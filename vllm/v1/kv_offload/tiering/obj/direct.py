# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side loads from the object store straight into GPU memory.

Two loaders read chunk objects with a NIXL OBJ backend whose S3-over-RDMA
engine writes GPU memory. With the accelerated engine and
``rdma_transport=ofi`` on Ceph, the OSDs RDMA-write the bytes into GPU
memory, so a load passes through neither the CPU tier nor the S3 endpoint.

ObjStagedLoader (the default) registers a ring of chunk-sized GPU staging
slots. Each chunk takes one GET of the worker's whole slice of the object
into a slot, and a device-to-device copy then scatters the pages into the KV
cache blocks while the next GETs land. GETs stay as large as a chunk, and
only the ring is registered with the NIC.

ObjDirectLoader registers the GPU KV cache itself and reads each block's
pages into place with ranged GETs. That needs no copy, but the KV cache keeps
one tensor per layer, so a chunk takes at least one GET per tensor.
"""

import os
import queue
import threading
import time
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm.distributed.nixl_utils import NixlWrapper as nixl_agent
from vllm.distributed.nixl_utils import nixl_agent_config
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import CanonicalKVCaches, TransferResult
from vllm.v1.kv_offload.tiering.direct_load import BlockPlacement
from vllm.v1.kv_offload.tiering.obj.config import ObjStoreConfig

logger = init_logger(__name__)

NIXL_READ = "READ"
NIXL_PROC = "PROC"
NIXL_DONE = "DONE"

# GPU memory is registered in regions of at most this size. Some RDMA NICs
# cap one memory registration (irdma on the Intel E810 at about 6 GiB).
DEFAULT_MAX_REGION_BYTES = 2 << 30

# GPU memory for ObjStagedLoader's staging slots.
DEFAULT_STAGING_BYTES = 1 << 30

# Staging slots start on this boundary.
_SLOT_ALIGNMENT = 1 << 16


def _backend_params(store_config: dict, io_threads: int) -> dict[str, str]:
    params = {
        **ObjStoreConfig(**store_config).to_nixl_params(),
        "num_threads": str(io_threads),
    }
    if params.get("accelerated", "").lower() != "true":
        raise ValueError(
            "gpu_direct_load needs an S3-over-RDMA backend that writes GPU "
            "memory: set store_config.nixl_params.accelerated to true, with "
            "an rdma_transport that registers GPU memory"
        )
    return params


def _open_agent(name: str, params: dict[str, str]) -> Any:
    agent = nixl_agent(name, nixl_agent_config(backends=[], capture_telemetry=True))
    agent.create_backend("OBJ", params)
    return agent


def _register_vram(
    agent: Any, regions: list[tuple[int, int]], device: int, what: str
) -> Any:
    try:
        return agent.register_memory(
            [(addr, size, device, "") for addr, size in regions], "VRAM"
        )
    except Exception as e:
        raise RuntimeError(
            f"gpu_direct_load could not register the {what} with the NIXL OBJ "
            "backend. The backend must support VRAM: an accelerated engine "
            "whose token provider registers GPU memory. With "
            "rdma_transport=ofi and a NIXL built without a GPU runtime, set "
            "nixl_params.ofi_hmem to rocr or cuda."
        ) from e


def _split_regions(
    base: int, total: int, unit: int, max_bytes: int
) -> list[tuple[int, int]]:
    """Regions of at most max_bytes covering [base, base + total), each a
    whole number of units, so that no unit straddles two regions."""
    step = max(unit, max_bytes // unit * unit)
    return [(base + start, min(step, total - start)) for start in range(0, total, step)]


class _KvLayout(NamedTuple):
    """Where a worker's KV cache pages sit, in GPU memory and in a chunk."""

    # Per canonical tensor: GPU base address, page size, and total bytes
    bases: np.ndarray
    pages: np.ndarray
    totals: np.ndarray
    # Per canonical tensor: offset of its pages in the worker's slice of a
    # chunk (SharedOffloadRegion concatenates the tensors' chunk pages)
    slice_offsets: np.ndarray
    # Size of the worker's slice of a chunk
    slice_bytes: int
    # Per KV group: the (tensor, bytes) pairs that one block occupies
    group_copies: list[list[tuple[int, int]]]

    @classmethod
    def build(cls, kv_caches: CanonicalKVCaches, blocks_per_chunk: int) -> "_KvLayout":
        bases, pages, totals, offsets = [], [], [], []
        offset = 0
        for kv_tensor in kv_caches.tensors:
            page = kv_tensor.page_size_bytes
            blocks = kv_tensor.tensor.view(torch.int8).view((-1, page))
            bases.append(blocks.data_ptr())
            pages.append(page)
            totals.append(blocks.shape[0] * page)
            offsets.append(offset)
            offset += page * blocks_per_chunk
        group_copies: list[list[tuple[int, int]]] = []
        for refs in kv_caches.group_data_refs:
            sizes: dict[int, int] = {}
            for ref in refs:
                sizes[ref.tensor_idx] = max(
                    sizes.get(ref.tensor_idx, 0), ref.page_size_bytes
                )
            group_copies.append(sorted(sizes.items()))
        return cls(
            bases=np.array(bases, dtype=np.int64),
            pages=np.array(pages, dtype=np.int64),
            totals=np.array(totals, dtype=np.int64),
            slice_offsets=np.array(offsets, dtype=np.int64),
            slice_bytes=offset,
            group_copies=group_copies,
        )


def _block_copies(
    bases: np.ndarray,
    pages: np.ndarray,
    offsets: np.ndarray,
    group_copies: list[list[tuple[int, int]]],
    block_ids: np.ndarray,
    placement: BlockPlacement,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One copy per (block, tensor): (chunk, source offset, GPU address, size)
    arrays. A tensor's pages start at offsets[tensor] in the source."""
    chunks, srcs, dsts, sizes = [], [], [], []
    block_ids = block_ids.astype(np.int64)
    for g_idx, copies in enumerate(group_copies):
        in_group = placement.group == g_idx
        if not in_group.any():
            continue
        blocks = block_ids[in_group]
        chunk = placement.chunk_idx[in_group]
        position = placement.position[in_group]
        for t_idx, size in copies:
            page = pages[t_idx]
            chunks.append(chunk)
            srcs.append(offsets[t_idx] + position * page)
            dsts.append(bases[t_idx] + blocks * page)
            sizes.append(np.full(len(blocks), size, dtype=np.int64))
    if not chunks:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty
    return (
        np.concatenate(chunks),
        np.concatenate(srcs),
        np.concatenate(dsts),
        np.concatenate(sizes),
    )


def _merge_runs(
    key: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    size: np.ndarray,
    boundary_starts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort copies by (key, src) and merge each copy into the previous one
    when both its source and destination follow on, with the same key and,
    if boundary_starts is given, in the same destination region."""
    if len(key) == 0:
        return key, src, dst, size
    order = np.lexsort((src, key))
    key, src, dst, size = (a[order] for a in (key, src, dst, size))
    follows = np.zeros(len(key), dtype=bool)
    follows[1:] = (
        (key[1:] == key[:-1])
        & (src[1:] == src[:-1] + size[:-1])
        & (dst[1:] == dst[:-1] + size[:-1])
    )
    if boundary_starts is not None:
        region = np.searchsorted(boundary_starts, dst, side="right")
        follows[1:] &= region[1:] == region[:-1]
    starts = np.flatnonzero(~follows)
    return key[starts], src[starts], dst[starts], np.add.reduceat(size, starts)


class _Transfer(NamedTuple):
    handle: Any
    obj_reg: Any
    nbytes: int


class ObjDirectLoader:
    """Reads chunks from the object store straight into GPU KV cache blocks."""

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        store_config: dict,
        blocks_per_chunk: int,
        chunk_bytes: int,
        worker_offset: int,
        io_threads: int = 4,
        max_region_bytes: int = DEFAULT_MAX_REGION_BYTES,
    ):
        """Args:
        kv_caches: The worker's canonical KV caches.
        store_config: The object tier's store_config (see ObjStoreConfig).
        blocks_per_chunk: GPU blocks per offloaded chunk.
        chunk_bytes: Size of a chunk object.
        worker_offset: Offset of this worker's pages in a chunk.
        io_threads: Number of NIXL I/O threads.
        max_region_bytes: Largest GPU memory region registered at once.

        """
        params = _backend_params(store_config, io_threads)
        self._chunk_bytes = chunk_bytes
        self._device = torch.accelerator.current_device_index()
        layout = _KvLayout.build(kv_caches, blocks_per_chunk)
        assert worker_offset + layout.slice_bytes <= chunk_bytes
        self._bases = layout.bases
        self._pages = layout.pages
        self._chunk_offsets = worker_offset + layout.slice_offsets
        self._group_copies = layout.group_copies
        regions = sorted(
            region
            for base, total, page in zip(layout.bases, layout.totals, layout.pages)
            for region in _split_regions(
                int(base), int(total), int(page), max_region_bytes
            )
        )
        self._region_starts = np.array([r[0] for r in regions], dtype=np.int64)

        self._name = f"ObjDirectAgent-{os.getpid()}"
        self._agent = _open_agent(self._name, params)
        self._gpu_reg = _register_vram(
            self._agent, regions, self._device, "GPU KV cache"
        )
        logger.info(
            "Direct object store loads into the KV cache: %d GPU regions, "
            "%.2f GiB, %d tensors",
            len(regions),
            sum(size for _, size in regions) / (1 << 30),
            len(layout.bases),
        )

        self._next_obj_dev_id = 1
        self._transfers: dict[int, _Transfer] = {}
        self._results: list[TransferResult] = []

    def _plan(
        self, block_ids: np.ndarray, placement: BlockPlacement
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ranged reads for the given blocks: (chunk, object offset, GPU
        address, size) arrays, with adjacent reads merged."""
        return _merge_runs(
            *_block_copies(
                self._bases,
                self._pages,
                self._chunk_offsets,
                self._group_copies,
                block_ids,
                placement,
            ),
            boundary_starts=self._region_starts,
        )

    def submit(
        self,
        job_id: int,
        names: Sequence[str],
        block_ids: np.ndarray,
        placement: BlockPlacement,
    ) -> bool:
        if len(block_ids) == 0:
            self._results.append(
                TransferResult(job_id=job_id, success=True, transfer_size=0)
            )
            return True
        chunk, offset, addr, size = self._plan(block_ids, placement)

        used = np.unique(chunk)
        dev_ids = np.arange(len(used), dtype=np.int64) + self._next_obj_dev_id
        self._next_obj_dev_id += len(used)
        dev_of_chunk = np.zeros(len(names), dtype=np.int64)
        dev_of_chunk[used] = dev_ids
        obj_reg = self._agent.register_memory(
            [
                (0, self._chunk_bytes, int(dev), names[int(c)])
                for c, dev in zip(used, dev_ids)
            ],
            "OBJ",
        )

        local = np.stack(
            [addr, size, np.full(len(addr), self._device, dtype=np.int64)], axis=1
        ).astype(np.uint64)
        remote = np.stack([offset, size, dev_of_chunk[chunk]], axis=1).astype(np.uint64)
        handle = None
        try:
            handle = self._agent.initialize_xfer(
                NIXL_READ,
                self._agent.get_xfer_descs(local, "VRAM"),
                self._agent.get_xfer_descs(remote, "OBJ"),
                self._name,
            )
            state = self._agent.transfer(handle)
        except Exception as e:
            logger.error("Direct load job %d could not start: %s", job_id, e)
            state = "ERR"
        if state == "ERR":
            if handle is not None:
                self._agent.release_xfer_handle(handle)
            self._agent.deregister_memory(obj_reg)
            self._results.append(TransferResult(job_id=job_id, success=False))
            return True
        nbytes = int(size.sum())
        self._transfers[job_id] = _Transfer(handle, obj_reg, nbytes)
        logger.debug(
            "Direct load job %d: %d blocks, %d chunks, %d reads, %d bytes",
            job_id,
            len(block_ids),
            len(used),
            len(addr),
            nbytes,
        )
        return True

    def _poll(self) -> None:
        for job_id, transfer in list(self._transfers.items()):
            try:
                state = self._agent.check_xfer_state(transfer.handle)
            except Exception as e:
                logger.error("Direct load job %d: state check failed: %s", job_id, e)
                state = "ERR"
            if state == NIXL_PROC:
                continue
            success = state == NIXL_DONE
            if not success:
                logger.error("Direct load job %d failed: %s", job_id, state)
            transfer_time = None
            if success:
                try:
                    telemetry = self._agent.get_xfer_telemetry(transfer.handle)
                    transfer_time = telemetry.xferDuration / 1e6
                except Exception as e:
                    logger.debug("no telemetry for direct load %d: %s", job_id, e)
            try:
                self._agent.release_xfer_handle(transfer.handle)
            except Exception as e:
                # The transfer may still write GPU memory: keep it pending.
                logger.warning("Direct load job %d: release failed: %s", job_id, e)
                continue
            try:
                self._agent.deregister_memory(transfer.obj_reg)
            except Exception as e:
                logger.warning("Direct load job %d: deregister failed: %s", job_id, e)
            del self._transfers[job_id]
            self._results.append(
                TransferResult(
                    job_id=job_id,
                    success=success,
                    transfer_size=(
                        transfer.nbytes if transfer_time is not None else None
                    ),
                    transfer_time=transfer_time,
                )
            )

    def get_finished(self) -> list[TransferResult]:
        self._poll()
        results, self._results = self._results, []
        return results

    def wait(self, job_ids: set[int]) -> None:
        while job_ids & self._transfers.keys():
            self._poll()
            time.sleep(0.0005)

    def shutdown(self) -> None:
        for job_id, transfer in self._transfers.items():
            try:
                self._agent.release_xfer_handle(transfer.handle)
                self._agent.deregister_memory(transfer.obj_reg)
            except Exception as e:
                logger.warning("Direct load job %d: cleanup failed: %s", job_id, e)
        self._transfers.clear()
        if self._gpu_reg is not None:
            try:
                self._agent.deregister_memory(self._gpu_reg)
            except Exception as e:
                logger.warning("failed to deregister the GPU KV cache: %s", e)
            self._gpu_reg = None


class _StagedJob(NamedTuple):
    job_id: int
    names: Sequence[str]
    block_ids: np.ndarray
    placement: BlockPlacement


class ObjStagedLoader:
    """Reads chunks into GPU staging slots, then copies them into the KV cache.

    A loader thread runs one job at a time. It splits the job's chunks into
    waves of half the slots. While one wave's GETs land in its half of the
    ring, the other wave is copied into the KV cache blocks on a side stream.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        store_config: dict,
        blocks_per_chunk: int,
        chunk_bytes: int,
        worker_offset: int,
        io_threads: int = 4,
        staging_bytes: int = DEFAULT_STAGING_BYTES,
        max_region_bytes: int = DEFAULT_MAX_REGION_BYTES,
    ):
        """Args:
        kv_caches: The worker's canonical KV caches.
        store_config: The object tier's store_config (see ObjStoreConfig).
        blocks_per_chunk: GPU blocks per offloaded chunk.
        chunk_bytes: Size of a chunk object.
        worker_offset: Offset of this worker's pages in a chunk.
        io_threads: Number of NIXL I/O threads.
        staging_bytes: GPU memory for staging slots. The loader uses at
            least two slots, each the size of the worker's slice of a chunk.
        max_region_bytes: Largest GPU memory region registered at once.

        """
        params = _backend_params(store_config, io_threads)
        self._chunk_bytes = chunk_bytes
        self._worker_offset = worker_offset
        self._device = torch.accelerator.current_device_index()
        self._layout = _KvLayout.build(kv_caches, blocks_per_chunk)
        assert worker_offset + self._layout.slice_bytes <= chunk_bytes
        self._slot_bytes = round_up(self._layout.slice_bytes, _SLOT_ALIGNMENT)
        self._wave = max(1, staging_bytes // self._slot_bytes // 2)
        self._staging = torch.empty(
            (2 * self._wave, self._slot_bytes),
            dtype=torch.int8,
            device=torch.device(
                torch.accelerator.current_accelerator().type, self._device
            ),
        )
        self._staging_base = self._staging.data_ptr()
        regions = _split_regions(
            self._staging_base,
            self._staging.numel(),
            self._slot_bytes,
            max_region_bytes,
        )

        self._name = f"ObjStagedAgent-{os.getpid()}"
        self._agent = _open_agent(self._name, params)
        self._staging_reg = _register_vram(
            self._agent, regions, self._device, "GPU staging slots"
        )
        logger.info(
            "Direct object store loads through GPU staging: %d slots of %.2f MiB "
            "(%.2f GiB), waves of %d chunks",
            2 * self._wave,
            self._slot_bytes / (1 << 20),
            self._staging.numel() / (1 << 30),
            self._wave,
        )

        self._next_obj_dev_id = 1
        self._jobs: queue.Queue[_StagedJob | None] = queue.Queue()
        self._lock = threading.Lock()
        self._results: list[TransferResult] = []
        self._pending: dict[int, threading.Event] = {}
        self._thread = threading.Thread(
            target=self._run, name="obj-staged-loader", daemon=True
        )
        self._thread.start()

    # --- planning ------------------------------------------------------

    def _copy_plan(
        self,
        chunk: np.ndarray,
        src_off: np.ndarray,
        dst: np.ndarray,
        size: np.ndarray,
        slot_of_chunk: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Device-to-device copies from the staging slots: (source address,
        destination address, size) arrays, with adjacent copies merged."""
        src = self._staging_base + slot_of_chunk[chunk] * self._slot_bytes + src_off
        _, src, dst, size = _merge_runs(np.zeros_like(src), src, dst, size)
        return src, dst, size

    # --- loader thread -------------------------------------------------

    def _start_wave(
        self, names: Sequence[str], chunks: np.ndarray, first_slot: int
    ) -> tuple[Any, Any]:
        n = len(chunks)
        dev_ids = np.arange(n, dtype=np.int64) + self._next_obj_dev_id
        self._next_obj_dev_id += n
        obj_reg = self._agent.register_memory(
            [
                (0, self._chunk_bytes, int(dev), names[int(c)])
                for c, dev in zip(chunks, dev_ids)
            ],
            "OBJ",
        )
        slots = first_slot + np.arange(n, dtype=np.int64)
        length = np.full(n, self._layout.slice_bytes, dtype=np.int64)
        local = np.stack(
            [
                self._staging_base + slots * self._slot_bytes,
                length,
                np.full(n, self._device, dtype=np.int64),
            ],
            axis=1,
        ).astype(np.uint64)
        remote = np.stack(
            [np.full(n, self._worker_offset, dtype=np.int64), length, dev_ids], axis=1
        ).astype(np.uint64)
        try:
            handle = self._agent.initialize_xfer(
                NIXL_READ,
                self._agent.get_xfer_descs(local, "VRAM"),
                self._agent.get_xfer_descs(remote, "OBJ"),
                self._name,
            )
        except Exception:
            self._agent.deregister_memory(obj_reg)
            raise
        if self._agent.transfer(handle) == "ERR":
            self._finish_wave(handle, obj_reg)
            raise RuntimeError("the object store reads could not start")
        return handle, obj_reg

    def _finish_wave(self, handle: Any, obj_reg: Any) -> str:
        """Wait for a wave's reads to stop, release them, and return the
        final state."""
        while (state := self._agent.check_xfer_state(handle)) == NIXL_PROC:
            time.sleep(0.0002)
        self._agent.release_xfer_handle(handle)
        self._agent.deregister_memory(obj_reg)
        return state

    def _copy(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        size: np.ndarray,
        stream: torch.cuda.Stream,
    ) -> None:
        """Copy within GPU memory on the side stream and wait for it."""
        from vllm import _custom_ops as ops

        with torch.cuda.stream(stream):
            ops.swap_blocks_batch(
                torch.from_numpy(src), torch.from_numpy(dst), torch.from_numpy(size)
            )
        stream.synchronize()

    def _load(self, job: _StagedJob, stream: torch.cuda.Stream) -> int:
        chunk, src_off, dst, size = _block_copies(
            self._layout.bases,
            self._layout.pages,
            self._layout.slice_offsets,
            self._layout.group_copies,
            job.block_ids,
            job.placement,
        )
        used = np.unique(chunk)
        waves = [used[i : i + self._wave] for i in range(0, len(used), self._wave)]
        in_flight: dict[int, tuple[Any, Any]] = {}
        try:
            for w in range(min(2, len(waves))):
                in_flight[w] = self._start_wave(
                    job.names, waves[w], (w % 2) * self._wave
                )
            for w, wave in enumerate(waves):
                state = self._finish_wave(*in_flight.pop(w))
                if state != NIXL_DONE:
                    raise RuntimeError(f"object store reads ended in state {state}")
                slot_of_chunk = np.zeros(len(job.names), dtype=np.int64)
                slot_of_chunk[wave] = (w % 2) * self._wave + np.arange(len(wave))
                mine = np.isin(chunk, wave)
                src, dst_w, size_w = self._copy_plan(
                    chunk[mine], src_off[mine], dst[mine], size[mine], slot_of_chunk
                )
                self._copy(src, dst_w, size_w, stream)
                if w + 2 < len(waves):
                    in_flight[w + 2] = self._start_wave(
                        job.names, waves[w + 2], (w % 2) * self._wave
                    )
        finally:
            # Leave no read writing into a slot that the next job reuses.
            for handle, obj_reg in in_flight.values():
                try:
                    self._finish_wave(handle, obj_reg)
                except Exception as e:
                    logger.warning(
                        "Staged load job %d: cleanup failed: %s", job.job_id, e
                    )
        return len(used) * self._layout.slice_bytes

    def _run(self) -> None:
        torch.accelerator.set_device_index(self._device)
        stream = torch.cuda.Stream()
        while (job := self._jobs.get()) is not None:
            start = time.monotonic()
            try:
                nbytes = self._load(job, stream)
                result = TransferResult(
                    job_id=job.job_id,
                    success=True,
                    transfer_size=nbytes,
                    transfer_time=time.monotonic() - start,
                )
            except Exception:
                logger.exception("Staged load job %d failed", job.job_id)
                result = TransferResult(job_id=job.job_id, success=False)
            with self._lock:
                self._results.append(result)
                self._pending.pop(job.job_id).set()

    # --- DirectLoader --------------------------------------------------

    def submit(
        self,
        job_id: int,
        names: Sequence[str],
        block_ids: np.ndarray,
        placement: BlockPlacement,
    ) -> bool:
        with self._lock:
            self._pending[job_id] = threading.Event()
        self._jobs.put(_StagedJob(job_id, names, block_ids, placement))
        return True

    def get_finished(self) -> list[TransferResult]:
        with self._lock:
            results, self._results = self._results, []
        return results

    def wait(self, job_ids: set[int]) -> None:
        with self._lock:
            events = [self._pending[j] for j in job_ids if j in self._pending]
        for event in events:
            event.wait()

    def shutdown(self) -> None:
        self._jobs.put(None)
        self._thread.join()
        if self._staging_reg is not None:
            try:
                self._agent.deregister_memory(self._staging_reg)
            except Exception as e:
                logger.warning("failed to deregister the GPU staging slots: %s", e)
            self._staging_reg = None
