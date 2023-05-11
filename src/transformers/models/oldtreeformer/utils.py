#!/usr/bin/env python3

import sys
from typing import Optional, Tuple
import math

import torch

def pprint(tree, N, H):
    start_idx = 0
    if math.log10(max(tree)) < 2:
        formatter = lambda x: f"{x:<2}"
        indenter = lambda i: " " * 2 * i
    else:
        formatter = lambda x: f"{x:<3}"
        indenter = lambda i: " " * 3 * i

    to_print = []
    for idx, row_len in enumerate(range(N, max(N - H, 0), -1)):
        row = tree[start_idx:start_idx+row_len]
        to_print.append((indenter(1)).join(map(formatter, row)))
        start_idx += row_len

    print(*to_print[::-1], sep='\n')


def inverse_tri(W: int, H: int):
    return ((2 * W) - H + (H ** 2)) // (2 * H)


def tri(N: int):
    return N * (N + 1) // 2


def get_tree_width(N: int, H: int):
    H = min(H, N)
    return tri(N) - tri(N - H)

def get_tree_base_from_width(W: int, H: int):
    inv = int((math.sqrt(8 * W + 1) - 1) / 2)
    if inv > H:
        lower_tri = tri(inv-H)
        Wp = W - lower_tri
        inv = int((math.sqrt(8 * Wp + 1) - 1) / 2)
        return inv
    return inv

def get_row_idxs(N: int, row: int) -> Tuple[int, int]:
    row_len = N - row
    start_idx = tri(N) - tri(row_len)
    end_idx = start_idx + row_len
    return start_idx, end_idx


def get_idx_from_coord(N: int, coord: Tuple[int, int]) -> int:
    row, col = coord
    start, end = get_row_idxs(N, row)
    return start + col

def get_coord_from_idx(N: int, idx):
    row = 0
    start, end = get_row_idxs(N, row)
    while end <= idx:
        row += 1
        start, end = get_row_idxs(N, row)
    col = idx - start
    return (row, col)


def make_tree_pad_mask(seq_mask: torch.Tensor, H: int, split_at: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Create padding mask for tree from base sequence

    Parameters
    ----------
    seq_mask : torch.Tensor, shape `[B, N]`
        Mask for some sequence, where `True` indicates padding

    Returns
    -------
    torch.Tensor, shape `[B, tri(N)]`:
        Padding mask for tree (`True` -> padding)
    """
    B, N = seq_mask.shape
    W = get_tree_width(N, H)
    tree_mask = torch.zeros(size=(B, W), device=seq_mask.device, dtype=torch.bool)
    tree_mask[:, :N] = seq_mask
    for row in range(1, min(N, H)):
        start, end = get_row_idxs(N, row)
        tree_mask[:, start:end] = seq_mask[:, row:]

    # TODO figure out how to make the mask. I think you can use get_row_idxs to get
    # [first_seq_start, first_seq_end] and [sec_seq_start, sec_seq_end], and then just
    # mask out the part between [first_seq_end, sec_seq_start]

    # NOTE Above has been solved I think. This logic should do it.

    if split_at is not None:
        for b in range(B):
            l1 = int(split_at[b].item())
            for row in range(1, min(N, H)):
                row_len_1 = l1 - row
                start1, end1 = get_row_idxs(N, row)

                # Mask out the in-between part
                tree_mask[b, start1+row_len_1:start1+l1] = True

    return tree_mask


def make_tree_attn_mask(
    N: int,
    H: int,
    attend_to_words: bool = False,
    attend_to_whole_tree: bool = False,
    additive: bool = False,
    **kwargs,
):
    """Construct tree attention mask for decoding.
    Arguments
    ---------
    N : torch.Tensor
        Sequence length
    H : torch.Tensor
        Tree height
    attend_to_words : bool
        If True, uses the bottom row as context
    attend_to_whole_tree : bool
        If True, uses the entire tree (excluding future) as context
    additive : bool
        If True, creates a mask with -inf over non-attended squares and 0 elsewhere
    kwargs : Dict
        Arguments to pass to tensor creation.
    Returns
    -------
    torch.Tensor :
        Mask of shape `[O, O(O+1)/2]`
    """
    W = get_tree_width(N, H)
    attn_mask = torch.zeros((N, W), dtype=torch.bool, **kwargs)
    for step in range(N):
        idxs = []
        if attend_to_whole_tree:
            for row in range(min(step + 1, H)):
                col = step - row
                start, end = get_row_idxs(N, row)
                attn_mask[step, start : start + col + 1] = 1
        else:
            for row in range(min(step + 1, H)):
                col = step - row
                idx = get_idx_from_coord(N, (row, col))
                idxs.append(idx)
            attn_mask[step, idxs] = 1
        if attend_to_words:
            attn_mask[step, :step] = 1

    if additive:
        output = torch.masked_fill(attn_mask.float(), ~attn_mask, float('-inf'))
        output = torch.masked_fill(output, attn_mask, 0.0)
        return output
    # Flip so True indicates padding
    return ~attn_mask


def assert_close(a: torch.Tensor, b: torch.Tensor, message: str):
    try:
        assert torch.allclose(a, b, atol=1e-5)
    except AssertionError:
        logger.exception(message)
        logger.error(f"Tensors differed by {abs(a - b).max()}")
        input("Press RET to print tensors...")
        logger.error(f"Tensor A: {a}")
        logger.error(f"Tensor B: {b}")
        sys.exit(1)
