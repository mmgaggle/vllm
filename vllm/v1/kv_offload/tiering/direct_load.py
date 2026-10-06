# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loads from a secondary tier straight into GPU memory.

A secondary tier with ``gpu_direct_load`` set does not promote a hit into the
CPU (primary) tier. The scheduler names the chunks in a ``DirectLoadSpec``,
and a worker-side ``DirectLoader`` reads them into the GPU KV cache blocks.
When one load mixes chunks from the primary tier and from such a tier, the
scheduler sends a ``TieredLoadSpec``: one segment per run of chunks with the
same source, in key order. ``TieringOffloadingWorker`` splits it into one job
per segment and reports the load done when every segment is done.

A stored chunk keeps the primary tier's layout (see SharedOffloadRegion):
for each canonical tensor, the chunk's GPU pages back to back, starting at
the worker's offset in the chunk.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

logger = init_logger(__name__)


class DirectLoadSpec(LoadStoreSpec):
    """Chunks that the workers read straight from a secondary tier into GPU
    memory, named by that tier's direct_load_names()."""

    def __init__(self, tier_idx: int, names: Sequence[str]):
        self.tier_idx = tier_idx
        self.names = list(names)

    def __repr__(self) -> str:
        return f"DirectLoadSpec(tier={self.tier_idx}, chunks={len(self.names)})"


class TieredLoadSpec(LoadStoreSpec):
    """A load whose chunks come from more than one source.

    Each segment is a CPULoadStoreSpec or a DirectLoadSpec covering the next
    run of chunks, in the order of the load's keys.
    """

    def __init__(self, segments: Sequence[LoadStoreSpec]):
        assert segments
        for segment in segments:
            assert isinstance(segment, (CPULoadStoreSpec, DirectLoadSpec))
        self.segments = list(segments)

    def __repr__(self) -> str:
        return f"TieredLoadSpec({self.segments!r})"


def segment_num_chunks(segment: LoadStoreSpec) -> int:
    if isinstance(segment, CPULoadStoreSpec):
        return len(segment.chunk_ids)
    assert isinstance(segment, DirectLoadSpec)
    return len(segment.names)


@dataclass(frozen=True)
class BlockPlacement:
    """Where each destination GPU block sits in the loaded chunks.

    All arrays have one entry per block of the GPULoadStoreSpec, in order.
    """

    # Index of the block's chunk among the load's keys
    chunk_idx: np.ndarray
    # Position of the block inside its chunk, in GPU blocks
    position: np.ndarray
    # KV cache group of the block
    group: np.ndarray
    # Number of chunks the load covers
    num_chunks: int


def place_blocks(dst_spec: GPULoadStoreSpec, blocks_per_chunk: int) -> BlockPlacement:
    """Map each destination block to its chunk and position in the chunk.

    Matches SingleDirectionOffloadingHandler: a group of n blocks starting at
    logical block L skips L % blocks_per_chunk blocks of its first chunk, and
    each group starts a new chunk.
    """
    chunk_idx: list[np.ndarray] = []
    position: list[np.ndarray] = []
    group: list[np.ndarray] = []
    chunk_base = 0
    for g_idx, (group_size, block_idx) in enumerate(
        zip(dst_spec.group_sizes, dst_spec.block_indices)
    ):
        if group_size == 0:
            continue
        skip = block_idx % blocks_per_chunk
        logical = skip + np.arange(group_size, dtype=np.int64)
        chunk_idx.append(chunk_base + logical // blocks_per_chunk)
        position.append(logical % blocks_per_chunk)
        group.append(np.full(group_size, g_idx, dtype=np.int64))
        chunk_base += cdiv(group_size + skip, blocks_per_chunk)

    def cat(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    return BlockPlacement(
        chunk_idx=cat(chunk_idx),
        position=cat(position),
        group=cat(group),
        num_chunks=chunk_base,
    )


def sub_gpu_spec(
    dst_spec: GPULoadStoreSpec, placement: BlockPlacement, lo: int, hi: int
) -> GPULoadStoreSpec:
    """The part of dst_spec whose blocks fall in chunks [lo, hi).

    Within each group those blocks are contiguous, so the result keeps the
    group structure that SingleDirectionOffloadingHandler expects.
    """
    block_ids: list[int] = []
    group_sizes: list[int] = []
    block_indices: list[int] = []
    offset = 0
    for group_size, block_idx in zip(dst_spec.group_sizes, dst_spec.block_indices):
        chunks = placement.chunk_idx[offset : offset + group_size]
        selected = np.flatnonzero((chunks >= lo) & (chunks < hi))
        if len(selected):
            first, last = int(selected[0]), int(selected[-1])
            assert last - first + 1 == len(selected)
            block_ids.extend(dst_spec.block_ids[offset + first : offset + last + 1])
            group_sizes.append(len(selected))
            block_indices.append(block_idx + first)
        else:
            group_sizes.append(0)
            block_indices.append(block_idx)
        offset += group_size
    return GPULoadStoreSpec(
        block_ids, group_sizes=group_sizes, block_indices=block_indices
    )


class DirectLoader(Protocol):
    """Worker-side reader for one gpu_direct_load tier."""

    def submit(
        self,
        job_id: int,
        names: Sequence[str],
        block_ids: np.ndarray,
        placement: BlockPlacement,
    ) -> bool:
        """Read chunk ``names[placement.chunk_idx[i]]`` position
        ``placement.position[i]`` into GPU block ``block_ids[i]``, for every
        i. Asynchronous: report completion through get_finished()."""
        ...

    def get_finished(self) -> list[TransferResult]: ...

    def wait(self, job_ids: set[int]) -> None: ...

    def shutdown(self) -> None: ...


@dataclass
class _ParentJob:
    pending: set[int]
    success: bool = True
    transfer_size: int = 0
    transfer_time: float = 0.0
    sizes_known: bool = True
    cpu_sub_ids: set[int] = field(default_factory=set)
    direct_sub_ids: dict[int, set[int]] = field(default_factory=dict)


class TieringOffloadingWorker(OffloadingWorker):
    """CPU offloading worker plus direct GPU loads from secondary tiers.

    Stores and plain CPU loads go to the CPU worker unchanged. A
    TieredLoadSpec becomes one sub-job per segment. Sub-jobs get negative
    ids, so they never collide with the connector's job ids.
    """

    def __init__(
        self,
        cpu_worker: OffloadingWorker,
        direct_loaders: dict[int, DirectLoader],
        blocks_per_chunk: int,
    ):
        self._cpu_worker = cpu_worker
        self._direct_loaders = direct_loaders
        self._blocks_per_chunk = blocks_per_chunk
        self._next_sub_id = -1
        self._parents: dict[int, _ParentJob] = {}
        self._parent_of: dict[int, int] = {}
        self._ready: list[TransferResult] = []

    def _new_sub_id(self) -> int:
        sub_id = self._next_sub_id
        self._next_sub_id -= 1
        return sub_id

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._cpu_worker.submit_store(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        if not isinstance(src_spec, TieredLoadSpec):
            return self._cpu_worker.submit_load(job_id, src_spec, dst_spec)

        placement = place_blocks(dst_spec, self._blocks_per_chunk)
        total = sum(segment_num_chunks(s) for s in src_spec.segments)
        if total != placement.num_chunks:
            logger.error(
                "Load job %d names %d chunks but its GPU blocks span %d",
                job_id,
                total,
                placement.num_chunks,
            )
            return False

        parent = _ParentJob(pending=set())
        self._parents[job_id] = parent
        lo = 0
        for segment in src_spec.segments:
            hi = lo + segment_num_chunks(segment)
            sub_id = self._new_sub_id()
            parent.pending.add(sub_id)
            self._parent_of[sub_id] = job_id
            if isinstance(segment, CPULoadStoreSpec):
                parent.cpu_sub_ids.add(sub_id)
                ok = self._cpu_worker.submit_load(
                    sub_id, segment, sub_gpu_spec(dst_spec, placement, lo, hi)
                )
            else:
                assert isinstance(segment, DirectLoadSpec)
                loader = self._direct_loaders.get(segment.tier_idx)
                if loader is None:
                    logger.error(
                        "Load job %d: tier %d has no direct loader",
                        job_id,
                        segment.tier_idx,
                    )
                    ok = False
                else:
                    parent.direct_sub_ids.setdefault(segment.tier_idx, set()).add(
                        sub_id
                    )
                    mask = (placement.chunk_idx >= lo) & (placement.chunk_idx < hi)
                    ok = loader.submit(
                        sub_id,
                        segment.names,
                        dst_spec.block_ids[mask],
                        BlockPlacement(
                            chunk_idx=placement.chunk_idx[mask] - lo,
                            position=placement.position[mask],
                            group=placement.group[mask],
                            num_chunks=hi - lo,
                        ),
                    )
            if not ok:
                # Report the failure with the parent job, once the sub-jobs
                # already submitted have finished with the memory they use.
                self._finish_sub(
                    TransferResult(job_id=sub_id, success=False), self._ready
                )
            lo = hi
        return True

    def _finish_sub(self, result: TransferResult, out: list[TransferResult]) -> None:
        parent_id = self._parent_of.pop(result.job_id)
        parent = self._parents[parent_id]
        parent.pending.discard(result.job_id)
        parent.success &= result.success
        if result.transfer_size is None or result.transfer_time is None:
            parent.sizes_known = False
        else:
            parent.transfer_size += result.transfer_size
            parent.transfer_time = max(parent.transfer_time, result.transfer_time)
        if parent.pending:
            return
        del self._parents[parent_id]
        out.append(
            TransferResult(
                job_id=parent_id,
                success=parent.success,
                transfer_size=parent.transfer_size if parent.sizes_known else None,
                transfer_time=parent.transfer_time if parent.sizes_known else None,
            )
        )

    def get_finished(self) -> list[TransferResult]:
        out, self._ready = self._ready, []
        sub_results = list(self._cpu_worker.get_finished())
        for loader in self._direct_loaders.values():
            sub_results.extend(loader.get_finished())
        for result in sub_results:
            if result.job_id in self._parent_of:
                self._finish_sub(result, out)
            else:
                out.append(result)
        return out

    def wait(self, job_ids: set[int]) -> None:
        cpu_ids = set()
        direct_ids: dict[int, set[int]] = {}
        for job_id in job_ids:
            parent = self._parents.get(job_id)
            if parent is None:
                cpu_ids.add(job_id)
                continue
            cpu_ids |= parent.cpu_sub_ids & parent.pending
            for tier_idx, sub_ids in parent.direct_sub_ids.items():
                direct_ids.setdefault(tier_idx, set()).update(sub_ids & parent.pending)
        if cpu_ids:
            self._cpu_worker.wait(cpu_ids)
        for tier_idx, sub_ids in direct_ids.items():
            if sub_ids:
                self._direct_loaders[tier_idx].wait(sub_ids)

    def shutdown(self) -> None:
        for loader in self._direct_loaders.values():
            loader.shutdown()
        self._cpu_worker.shutdown()
