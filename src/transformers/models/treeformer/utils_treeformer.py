#!/usr/bin/env python3
#!/usr/bin/env python3

import math
from functools import lru_cache
from typing import Tuple, Union

import torch
import numpy as np


def pprint_tree(tree):
    start = 0
    to_print = []
    N = (len(tree) + 1) // 2
    row_lens = get_row_lens(N)
    for r, row_len in enumerate(row_lens):
        row = tree[start : start + row_len]
        to_print.append(" ".join(map(lambda x: f"{x:<2}", map(int, row))))
        start += row_len
    print(*to_print[::-1], sep="\n")


def tree_size(N: int) -> int:
    return 2 * N - 1


def tree_height(N: int) -> int:
    return math.ceil(math.log(2 * N, 2))


def get_row_lens(N: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    lens = torch.zeros(tree_height(N), dtype=torch.long, device=device)
    remainders = 0
    row_len = int(N)
    idx = 0
    while row_len:
        lens[idx] = row_len
        remainders += row_len % 2
        row_len >>= 1
        if remainders % 2 == 0:
            row_len += remainders // 2
            remainders = 0
        idx += 1
    return lens


def get_row_idxs(N: int, rows: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    row_lens = get_row_lens(N, device=device)
    ends = torch.cumsum(row_lens, dim=0)
    starts = ends - row_lens
    return torch.stack([starts[rows], ends[rows]], dim=0).T


def get_coords_from_idxs(N: int, idxs: torch.Tensor) -> torch.Tensor:
    row_lens = get_row_lens(N, device=idxs.device)
    ends = torch.cumsum(row_lens, dim=0)
    starts = ends - row_lens
    rows = torch.nonzero((ends > idxs.view(-1, 1)) & (starts <= idxs.view(-1, 1)))[:, 1]
    cols = idxs - starts[rows]
    return torch.stack([rows, cols], dim=1)


def get_idxs_from_coords(N: int, coords: torch.Tensor) -> torch.Tensor:
    rows = coords[:, 0]
    cols = coords[:, 1]
    starts, ends = get_row_idxs(N, rows)
    return starts + cols


def make_tree_pad_mask(seq_mask: np.ndarray) -> np.ndarray:
    """Convert sequence pad mask to tree pad mask"""
    B, N = seq_mask.shape
    W = tree_size(N)
    mask = np.zeros((B, W), dtype=bool)
    seq_lengths = np.sum(seq_mask == 0, axis=1)

    for b, L in enumerate(seq_lengths):
        row_lens = get_row_lens(L)
        starts = get_row_idxs(N, torch.arange(tree_height(N)))[:, 0]
        for row, row_len in enumerate(row_lens):
            start = starts[row]
            mask[b, start : start + row_len] = seq_mask[b, :row_len]

    return mask


@lru_cache
def make_tree_attn_mask(N):
    """Generate the attention mask for a sequence of length `N` padded to length `batch_max_N`

    A cell in row `r` can attend to any cell in row `t <= r`.
    """
    W = tree_size(N)
    mask = np.zeros((W, W), dtype=bool)
    starts_and_ends = get_row_idxs(N, torch.arange(tree_height(N)))
    for row in range(tree_height(N)):
        start = starts_and_ends[row, 0].item()
        end = starts_and_ends[row, 1].item()
        mask[start:, start:end] = 1  # allow attention
    return mask


def tree_masks(mask, device: torch.device = torch.device("cuda")):
    """Builds tuple of `(attn_mask, pad_mask)`"""

    # convert to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # generate masks (in numpy)
    attn_mask = make_tree_attn_mask(mask.shape[1])[None, :, :]
    pad_mask = make_tree_pad_mask(mask)[:, :, None]

    # convert to torch
    attn_mask = torch.tensor(attn_mask, device=device)
    pad_mask = torch.tensor(pad_mask, device=device)

    # convert to additive mask
    attn_mask = attn_mask & pad_mask & pad_mask.permute(0, 2, 1)
    # attn_mask = (~attn_mask).float() * -1e10
    return (attn_mask, pad_mask.squeeze(-1))


def random_mask(B, L):
    """Generate a random padding mask for testing."""
    mask = np.zeros((B, L), dtype=np.int32)
    lens = np.random.randint(max(L - 5, 3), L + 1, (B,))
    for b, l in enumerate(lens):
        mask[b, :l] = 1
    return mask
