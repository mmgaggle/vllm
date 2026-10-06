# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side loads from the object store straight into GPU memory.

ObjDirectLoader registers the GPU KV cache with a NIXL OBJ backend and reads
each loaded block's bytes from its chunk object with a ranged GET into the
block's GPU pages. With the accelerated engine and ``rdma_transport=ofi`` on
Ceph, the OSDs RDMA-write those bytes into GPU memory, so a load from the
object store passes through neither the CPU tier nor the S3 endpoint.

Adjacent pages that are adjacent in both the object and GPU memory share one
GET: a chunk loaded into consecutive GPU blocks takes one GET per canonical
tensor.
"""

import os
import time
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm.distributed.nixl_utils import NixlWrapper as nixl_agent
from vllm.distributed.nixl_utils import nixl_agent_config
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import CanonicalKVCaches, TransferResult
from vllm.v1.kv_offload.tiering.direct_load import BlockPlacement
from vllm.v1.kv_offload.tiering.obj.config import ObjStoreConfig

logger = init_logger(__name__)

NIXL_READ = "READ"
NIXL_PROC = "PROC"
NIXL_DONE = "DONE"

# The GPU KV cache is registered in regions of at most this size. Some RDMA
# NICs cap one memory registration (irdma on the Intel E810 at about 6 GiB).
DEFAULT_MAX_REGION_BYTES = 2 << 30


class _Transfer(NamedTuple):
    handle: Any
    obj_reg: Any
    nbytes: int


class ObjDirectLoader:
    """Reads chunks from the object store into GPU KV cache blocks."""

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
        self._blocks_per_chunk = blocks_per_chunk
        self._chunk_bytes = chunk_bytes
        self._device = torch.accelerator.current_device_index()

        # Per canonical tensor: base address, page size, offset in a chunk.
        bases, pages, chunk_offsets, regions = [], [], [], []
        offset = worker_offset
        for kv_tensor in kv_caches.tensors:
            page = kv_tensor.page_size_bytes
            blocks = kv_tensor.tensor.view(torch.int8).view((-1, page))
            base = blocks.data_ptr()
            total = blocks.shape[0] * page
            bases.append(base)
            pages.append(page)
            chunk_offsets.append(offset)
            offset += page * blocks_per_chunk
            # Regions hold whole pages, so no page straddles two regions.
            step = max(page, max_region_bytes // page * page)
            for start in range(0, total, step):
                regions.append((base + start, min(step, total - start)))
        assert offset <= chunk_bytes, (offset, chunk_bytes)
        self._bases = np.array(bases, dtype=np.int64)
        self._pages = np.array(pages, dtype=np.int64)
        self._chunk_offsets = np.array(chunk_offsets, dtype=np.int64)
        regions.sort()
        self._region_starts = np.array([r[0] for r in regions], dtype=np.int64)

        # Per KV group: the (tensor, bytes) pairs that one block occupies.
        self._group_copies: list[list[tuple[int, int]]] = []
        for refs in kv_caches.group_data_refs:
            sizes: dict[int, int] = {}
            for ref in refs:
                sizes[ref.tensor_idx] = max(
                    sizes.get(ref.tensor_idx, 0), ref.page_size_bytes
                )
            self._group_copies.append(sorted(sizes.items()))

        self._name = f"ObjDirectAgent-{os.getpid()}"
        self._agent = nixl_agent(
            self._name, nixl_agent_config(backends=[], capture_telemetry=True)
        )
        self._agent.create_backend("OBJ", params)
        try:
            self._gpu_reg = self._agent.register_memory(
                [(addr, size, self._device, "") for addr, size in regions], "VRAM"
            )
        except Exception as e:
            raise RuntimeError(
                "gpu_direct_load could not register the GPU KV cache with the "
                "NIXL OBJ backend. The backend must support VRAM: an accelerated "
                "engine whose token provider registers GPU memory. With "
                "rdma_transport=ofi and a NIXL built without a GPU runtime, set "
                "nixl_params.ofi_hmem to rocr or cuda."
            ) from e
        logger.info(
            "Direct object store loads: %d GPU regions, %.2f GiB, %d tensors",
            len(regions),
            sum(size for _, size in regions) / (1 << 30),
            len(bases),
        )

        self._next_obj_dev_id = 1
        self._transfers: dict[int, _Transfer] = {}
        self._results: list[TransferResult] = []

    def _plan(
        self, block_ids: np.ndarray, placement: BlockPlacement
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ranged reads for the given blocks: (chunk, object offset, GPU
        address, size) arrays, with adjacent reads merged."""
        chunks, offsets, addrs, sizes = [], [], [], []
        block_ids = block_ids.astype(np.int64)
        for g_idx, copies in enumerate(self._group_copies):
            in_group = placement.group == g_idx
            if not in_group.any():
                continue
            blocks = block_ids[in_group]
            chunk = placement.chunk_idx[in_group]
            position = placement.position[in_group]
            for t_idx, size in copies:
                page = self._pages[t_idx]
                chunks.append(chunk)
                offsets.append(self._chunk_offsets[t_idx] + position * page)
                addrs.append(self._bases[t_idx] + blocks * page)
                sizes.append(np.full(len(blocks), size, dtype=np.int64))
        chunk = np.concatenate(chunks)
        offset = np.concatenate(offsets)
        addr = np.concatenate(addrs)
        size = np.concatenate(sizes)

        order = np.lexsort((offset, chunk))
        chunk, offset, addr, size = (a[order] for a in (chunk, offset, addr, size))
        region = np.searchsorted(self._region_starts, addr, side="right")
        follows = np.zeros(len(chunk), dtype=bool)
        follows[1:] = (
            (chunk[1:] == chunk[:-1])
            & (offset[1:] == offset[:-1] + size[:-1])
            & (addr[1:] == addr[:-1] + size[:-1])
            & (region[1:] == region[:-1])
        )
        starts = np.flatnonzero(~follows)
        return (
            chunk[starts],
            offset[starts],
            addr[starts],
            np.add.reduceat(size, starts),
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
