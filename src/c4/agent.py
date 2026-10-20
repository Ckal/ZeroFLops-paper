"""Policy wrapper around the V6 GF(256) byte-LM for Connect Four.

The V6 model is a next-BYTE model: it sees a token stream and returns logits
over all 256 byte values. Nothing about it needs to change for Connect Four.
We feed it the 10-byte-per-ply training stream and read the logits at the final
position, slicing the seven column bytes (0x41..0x47 = 'A'..'G'). Only those
seven values are ever legal actions, so the 256-way head is an over-complete
readout for a 7-way decision -- and the over-completeness is exactly what makes
the hardened integer export work unchanged.

Keeping the stock 256-way head means `export_hard()` / `hard_forward()` /
`generate.py`'s GFCipher all keep working with no surgery at all.
"""

from __future__ import annotations

import random
from typing import List, Sequence

import torch

from . import COLS
from .board import Board
from .transcript import PLY_LEN, COL_BYTE_OFFSET

N_COL_BYTES = COLS                     # 7
COL_SLICE = slice(COL_BYTE_OFFSET, COL_BYTE_OFFSET + N_COL_BYTES)

# Left-pad byte for hardened inference. 0x00 never appears in a corpus line,
# so a padded position is one the model has provably never been trained on --
# its contribution to the readout is whatever the untrained behaviour is, and
# the actual last-ply read-out at index -1 is unaffected.
PAD_BYTE = 0x00

# V6's hard_forward shifts by max(lags) = 64, so any prompt shorter than that
# collapses the shifted slice. Match V6's training seqlen so hard and soft see
# the same window the model was fit on.
HARD_MIN_LEN = 96


def moves_to_tokens(moves: Sequence[int]) -> torch.Tensor:
    """Encode a move history into the model's input token stream.

    Returns shape (1, T) of int64 byte values. The trailing newline of
    encode_game is stripped so the last token is a real ply byte.

    The prompt ends on the *turn* byte, not the move byte. Training is plain
    next-byte prediction over the flat corpus, so the logits at index i are
    trained to predict byte i+1. A ply record is
    [h0..h6, last, turn, move], and only the byte right after `turn` is the
    move we want. Ending the prompt at the move byte instead would read out
    the model's guess of the next ply's first height byte -- a value that
    looks plausible and is never a legal move.
    """
    from .transcript import encode_game

    line = encode_game(list(moves))
    return torch.tensor([list(line[:-2])], dtype=torch.long)


def start_of_game_tokens() -> torch.Tensor:
    """The empty-board position: all heights 0, no last move, P1 to play.

    encode_game([]) is just a newline, which gives a zero-width prompt. The
    first ply record of an empty board is
    [0x30 x7, NO_LAST, TOK_X, move] -- everything except the move is known
    before P1 has chosen, so it is a well-defined prefix to condition on.
    """
    from .transcript import COL_BYTE_OFFSET, HEIGHT_OFFSET, NO_LAST, TOK_X

    head = bytes([HEIGHT_OFFSET] * 7) + bytes((NO_LAST, TOK_X))
    return torch.tensor([list(head)], dtype=torch.long)


def prompt_tokens(moves: Sequence[int]) -> torch.Tensor:
    """Token stream to condition on for choosing the next move."""
    if len(moves) == 0:
        return start_of_game_tokens()
    return moves_to_tokens(moves)


def hard_to_cpu(hard):
    """Move a whole export_hard() dict to CPU, recursively.

    export_hard() hands back tensors on the model's device (CUDA), but
    hard_forward() rebuilds its EXP/LOG tables with torch.tensor(...) on CPU.
    Mixing the two makes `log_t[src]` index a CPU table with CUDA indices.
    Keeping V6 untouched means moving the export instead, which is free: the
    hardened path is already pure integer table lookups, so CPU is the right
    home for it anyway.
    """
    if torch.is_tensor(hard):
        return hard.cpu()
    if isinstance(hard, dict):
        return {k: hard_to_cpu(v) for k, v in hard.items()}
    if isinstance(hard, (list, tuple)):
        return type(hard)(hard_to_cpu(v) for v in hard)
    return hard


class Agent:
    """Wraps a GFMarkovLM with Connect-Four-legal move selection."""

    def __init__(self, model, device: str | torch.device = "cuda",
                 greedy: bool = True, temperature: float = 1.0,
                 rng: random.Random | None = None):
        self.model = model
        self.device = device
        self.greedy = greedy
        self.temperature = max(1e-6, temperature)
        self.hard_min_len = HARD_MIN_LEN
        self.rng = rng or random.Random(0)

    # ---- scoring ----

    @torch.no_grad()
    def column_logits(self, moves: Sequence[int]) -> torch.Tensor:
        """Logits for the seven column bytes at the position after `moves`.

        Shape (7,). The board implied by `moves` is the model's context; the
        returned entry j is the score for column j.
        """
        toks = prompt_tokens(moves).to(self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            logits = self.model(toks)[0, -1]      # (256,)
        finally:
            self.model.train(was_training)
        return logits.float()[COL_SLICE].cpu()

    @torch.no_grad()
    def batch_column_logits(self, batch_moves: Sequence[Sequence[int]],
                            ) -> torch.Tensor:
        """Column logits for many positions at once. Shape (N, 7).

        Used by self-play: one forward pass covers N positions, which is the
        difference between hours and minutes of corpus generation.
        """
        seqs = [prompt_tokens(m) for m in batch_moves]
        T = max(s.shape[1] for s in seqs)
        # Left-pad so every row ends on its own final turn byte; padding is
        # a byte value the corpus never produces in a meaningful position, and
        # the read-out is at index -1 regardless of padding.
        rows = []
        for s in seqs:
            pad = T - s.shape[1]
            rows.append(torch.cat([torch.full((1, pad), 0x00, dtype=torch.long),
                                   s], dim=1))
        x = torch.cat(rows, 0).to(self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            logits = self.model(x)[:, -1]        # (N, 256)
        finally:
            self.model.train(was_training)
        return logits.float()[:, COL_SLICE].cpu()

    @torch.no_grad()
    def hard_column_logits(self, moves: Sequence[int], hard) -> torch.Tensor:
        """Same as column_logits but through the hardened integer forward.

        The GF body (embedding, XOR convolution, sboxes) runs as pure table
        lookups and integer XORs; only the float readout stays float, which is
        what the exported model actually is. This runs on CPU on purpose: the
        hardened path has no matmuls to accelerate, so the GPU buys nothing
        and only risks a device mismatch with hard_forward's internal tables.

        The prompt is left-padded to `hard_min_len` bytes. V6's hard_forward
        shifts by the largest configured lag (64), and a sequence shorter than
        that collapses `src[:, :-64]` to zero width while the zero-pad it
        concatenates stays 64 wide. Training always runs at 96 tokens so the
        bug never fires there; a 40-byte Connect-Four prompt does hit it.
        Padding with a byte value the corpus never emits keeps the readout
        at index -1 exactly where the soft path reads it.
        """
        cpu_hard = hard_to_cpu(hard)
        cpu_toks = prompt_tokens(moves)
        pad = self.hard_min_len - cpu_toks.shape[1]
        if pad > 0:
            cpu_toks = torch.cat([torch.full((1, pad), PAD_BYTE, dtype=torch.long),
                                  cpu_toks], dim=1)
        was_training = self.model.training
        was_ste = self.model.ste
        self.model.eval()
        self.model.set_ste(False)
        try:
            cpu_logits = self.model.hard_forward(cpu_toks, cpu_hard)[0, -1]
        finally:
            self.model.set_ste(was_ste)
            self.model.train(was_training)
        return cpu_logits.float()[COL_SLICE].cpu()

    # ---- policy ----

    def legal_mask(self, board: Board) -> torch.Tensor:
        """-inf on illegal columns, 0 elsewhere. Shape (7,)."""
        m = torch.zeros(N_COL_BYTES)
        for c in range(N_COL_BYTES):
            if c not in board.valid_actions():
                m[c] = float("-inf")
        return m

    def choose(self, moves: List[int], board: Board) -> int:
        """Pick a legal column.

        Greedy by default (argmax over legal columns). With greedy=False it
        samples from the softmax over legal columns, which is what diversifies
        self-play -- an always-greedy self-player converges to one line.
        """
        scores = self.column_logits(moves) + self.legal_mask(board)
        if self.greedy:
            return int(scores.argmax())
        p = torch.softmax(scores / self.temperature, dim=-1)
        # torch.multinomial needs float probabilities
        return int(torch.multinomial(p, 1).item())

    def choose_batch(self, batch_moves: Sequence[Sequence[int]],
                     boards: Sequence[Board]) -> List[int]:
        """Vectorised choose() for self-play. Falls back to a one-sided batch
        (the opponent's move) rather than splitting it."""
        if len(batch_moves) == 1:
            return [self.choose(list(batch_moves[0]), boards[0])]
        scores = self.batch_column_logits(batch_moves)
        picks = []
        for i, (mv, b) in enumerate(zip(batch_moves, boards)):
            s = scores[i] + self.legal_mask(b)
            if self.greedy:
                picks.append(int(s.argmax()))
            else:
                p = torch.softmax(s / self.temperature, dim=-1)
                picks.append(int(torch.multinomial(p, 1).item()))
        return picks


def agent_agreement(agent: Agent, moves: Sequence[int], board: Board,
                    hard) -> tuple[bool, int]:
    """(soft_move == hard_move, soft_move) at this position.

    This is the only metric that says whether the hardened runtime will behave
    like the trained one; everything else is floating-point agreement.
    """
    soft = agent.choose(list(moves), board)
    hl = agent.hard_column_logits(moves, hard) + agent.legal_mask(board)
    hard_move = int(hl.argmax())
    return soft == hard_move, soft