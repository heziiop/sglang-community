"""Bounded NPU gather buffer for the Ascend DCP 1 -> N relayout.

A prefill rank running DCP=1 holds the whole logical token sequence, while
every decode DCP rank owns only its own residue of it. The rank-local attention
KV therefore has to be re-gathered per destination rank before it can be sent.
The staging area cannot be sized by the request or by the maximum prefill
length: it is allocated once at bootstrap and reused by every request that
lands on its transfer queue.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Sequence

import numpy as np
import numpy.typing as npt
import torch

logger = logging.getLogger(__name__)


class AscendDCPPackBuffer:
    """One bounded pack region, owned by a single transfer queue.

    Every local entry gets a fixed region of ``batch_indices`` token rows, so a
    short final batch only uses a prefix of each region and the region bases
    never move. The contents stay valid until the RDMA reading them has
    completed: the queue is serial, and `batch_transfer_sync()` returns before
    the next batch reuses the buffer.
    """

    def __init__(
        self,
        *,
        device: str,
        batch_indices: int,
        entry_token_bytes: Sequence[int],
    ):
        self._batch_indices = batch_indices
        self._entry_offsets: List[int] = []
        size_bytes = 0
        for token_bytes in entry_token_bytes:
            self._entry_offsets.append(size_bytes)
            size_bytes += batch_indices * token_bytes
        self._buffer = torch.zeros(size_bytes, dtype=torch.uint8, device=device)
        self._row_indices = torch.empty(batch_indices, dtype=torch.int64, device=device)
        self._gather_stream = torch.npu.Stream()
        self._gather_done = torch.npu.Event()

    def get_ptr(self) -> int:
        return self._buffer.data_ptr()

    def get_size(self) -> int:
        return self._buffer.numel()

    def pack(
        self,
        *,
        entries: Sequence[int],
        src_tensors: Sequence[torch.Tensor],
        token_item_lens: Sequence[int],
        src_token_indices: npt.NDArray[np.int64],
    ) -> List[int]:
        """Gather one local batch; return the packed source pointer per entry.

        `entries` are local KV entry indices, in the same order as the entry
        token sizes this buffer was built with. The regular transfer path has
        already established source-KV readiness before enqueueing the work. Keep
        the index upload and gather together on this buffer's private stream,
        and only return once the packed rows have landed.
        """
        count = int(src_token_indices.size)
        if count == 0:
            return []
        if count > self._batch_indices:
            raise RuntimeError(
                "Ascend DCP pack batch exceeds the buffer: "
                f"tokens={count}, batch_indices={self._batch_indices}"
            )
        row_indices = self._row_indices[:count]
        host_row_indices = torch.from_numpy(
            np.ascontiguousarray(src_token_indices, dtype=np.int64)
        )
        packed_ptrs: List[int] = []
        with torch.npu.stream(self._gather_stream):
            row_indices.copy_(host_row_indices)
            for slot, entry in enumerate(entries):
                token_bytes = int(token_item_lens[entry])
                region = self._region(slot, count, token_bytes)
                # Byte rows: the pool mixes dtypes (bf16, FP8, FP32 scales) and
                # `index_select` on FP8 is not uniformly supported on NPU.
                src_rows = src_tensors[entry].view(torch.uint8).reshape(-1, token_bytes)
                torch.index_select(src_rows, 0, row_indices, out=region)
                packed_ptrs.append(self._buffer.data_ptr() + self._entry_offsets[slot])
            # Event synchronization waits until the record task has passed
            # through torch_npu's task queue before waiting for NPU completion.
            self._gather_done.record(self._gather_stream)
        self._gather_done.synchronize()
        return packed_ptrs

    def _region(self, slot: int, count: int, token_bytes: int) -> torch.Tensor:
        return self._buffer.narrow(
            0, self._entry_offsets[slot], count * token_bytes
        ).view(count, token_bytes)


def init_ascend_dcp_pack_buffers(
    register_fn: Callable[[List[int], List[int]], None],
    *,
    batch_indices: int,
    page_size: int,
    kv_item_lens: Sequence[int],
    local_entry_indices: Sequence[int],
    count: int,
    device: str,
) -> List[AscendDCPPackBuffer]:
    """Allocate and register one pack buffer per transfer queue.

    Sized by the batch limit alone, never by the request or the maximum prefill
    length. All buffers are registered in one call because MemFabric aligns
    every region to 2 MiB: registering small neighbouring tensors one at a time
    publishes overlapping aligned ranges.
    """
    if batch_indices <= 0:
        raise RuntimeError(
            "Ascend PD DCP relayout requires "
            "SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES > 0 on the prefill server"
        )
    if len(set(local_entry_indices)) != len(local_entry_indices) or any(
        entry < 0 or entry >= len(kv_item_lens) for entry in local_entry_indices
    ):
        raise RuntimeError(
            "Ascend PD DCP relayout received invalid rank-local KV entry indices: "
            f"entries={list(local_entry_indices)}, item_lens={len(kv_item_lens)}"
        )
    if any(item_len % page_size for item_len in kv_item_lens):
        raise RuntimeError(
            f"Ascend PD DCP relayout needs page-aligned KV entries, page_size={page_size}"
        )
    local_entry_token_bytes = [
        kv_item_lens[entry] // page_size for entry in local_entry_indices
    ]
    if not local_entry_token_bytes:
        raise RuntimeError(
            "Ascend PD DCP relayout found no rank-local KV entry to pack; "
            "the target MLA attention KV must use DCP rank-local slots"
        )
    buffers = [
        AscendDCPPackBuffer(
            device=device,
            batch_indices=batch_indices,
            entry_token_bytes=local_entry_token_bytes,
        )
        for _ in range(count)
    ]
    register_fn(
        [buffer.get_ptr() for buffer in buffers],
        [buffer.get_size() for buffer in buffers],
    )
    logger.info(
        "Ascend DCP pack buffers: queues=%d batch_indices=%d local_entries=%d "
        "bytes_per_buffer=%d total_bytes=%d",
        count,
        batch_indices,
        len(local_entry_token_bytes),
        buffers[0].get_size(),
        sum(buffer.get_size() for buffer in buffers),
    )
    return buffers
