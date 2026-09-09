from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt


def build_dcp_replicated_kv_entry_mask(
    num_entries: int,
    *,
    replicated_group: int,
    group_count: int,
    num_draft_layers: int = 0,
) -> List[bool]:
    """Select full-copy entries in the NPU target-then-draft KV layout.

    The target indexer group is replicated, while every draft KV group is
    replicated because draft attention consumes allocator-global slot ids.
    """
    if num_entries < 0 or group_count <= 0 or num_draft_layers < 0:
        raise ValueError(
            "Ascend KV layout sizes must be non-negative with positive groups, "
            f"got entries={num_entries}, groups={group_count}, "
            f"draft_layers={num_draft_layers}"
        )
    if not 0 <= replicated_group < group_count:
        raise ValueError(
            "Ascend replicated KV group is outside the grouped layout: "
            f"replicated_group={replicated_group}, groups={group_count}"
        )

    draft_entries = group_count * num_draft_layers
    target_entries = num_entries - draft_entries
    if target_entries < 0 or target_entries % group_count:
        raise ValueError(
            "Ascend KV entries do not match the target/draft grouped layout: "
            f"entries={num_entries}, groups={group_count}, "
            f"draft_layers={num_draft_layers}"
        )

    target_layers = target_entries // group_count
    return [
        group_id == replicated_group
        for group_id in range(group_count)
        for _ in range(target_layers)
    ] + [True] * draft_entries


def build_replicated_dcp_token_indices(
    src_page_indices: npt.NDArray[np.int32],
    dst_page_indices: npt.NDArray[np.int32],
    *,
    physical_page_size: int,
    dcp_size: int,
    src_page_offset: int = 0,
    num_kv_tokens: Optional[int] = None,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Map every source token to the decode allocator's widened slot space."""
    if physical_page_size <= 0 or dcp_size <= 0:
        raise ValueError(
            "Ascend PD DCP page and topology sizes must be positive, got "
            f"page_size={physical_page_size}, dcp_size={dcp_size}"
        )
    if src_page_offset < 0:
        raise ValueError(
            "Ascend PD DCP source page offset must be non-negative, got "
            f"{src_page_offset}"
        )

    src_pages = np.asarray(src_page_indices, dtype=np.int64)
    dst_pages = np.asarray(dst_page_indices, dtype=np.int64)
    source_capacity = src_pages.size * physical_page_size
    if num_kv_tokens is None:
        num_kv_tokens = source_capacity
    if not 0 <= num_kv_tokens <= source_capacity:
        raise ValueError(
            "num_kv_tokens must fit in the provided Ascend source pages, "
            f"got tokens={num_kv_tokens}, capacity={source_capacity}"
        )
    if src_pages.size == 0 or num_kv_tokens == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy()

    offsets = np.arange(num_kv_tokens, dtype=np.int64)
    src_token_indices = (
        src_pages[offsets // physical_page_size] * physical_page_size
        + offsets % physical_page_size
    )

    virtual_page_size = physical_page_size * dcp_size
    relative_positions = src_page_offset * physical_page_size + offsets
    dst_page_ordinals = relative_positions // virtual_page_size
    if dst_pages.size == 0 or int(dst_page_ordinals.max()) >= dst_pages.size:
        required_pages = int(dst_page_ordinals.max()) + 1
        raise ValueError(
            "Insufficient destination Ascend DCP pages: "
            f"required={required_pages}, provided={dst_pages.size}, "
            f"src_page_offset={src_page_offset}"
        )
    dst_token_indices = (
        dst_pages[dst_page_ordinals] * virtual_page_size
        + relative_positions % virtual_page_size
    )
    return src_token_indices, dst_token_indices
