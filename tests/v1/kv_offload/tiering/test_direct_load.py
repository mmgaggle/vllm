# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for direct GPU loads from a secondary tier."""

import ctypes
from collections.abc import Collection, Iterable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LookupResult,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    TransferResult,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.tiering.direct_load import (
    BlockPlacement,
    DirectLoadSpec,
    TieredLoadSpec,
    TieringOffloadingWorker,
    place_blocks,
    sub_gpu_spec,
)
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.obj.direct import (
    ObjDirectLoader,
    ObjStagedLoader,
    _KvLayout,
    _StagedJob,
)


def to_keys(int_ids: Iterable[int]) -> list[OffloadKey]:
    return [make_offload_key(str(i).encode(), 0) for i in int_ids]


def _mock_mmap_region(num_chunks: int, row_bytes: int = 16):
    mock = MagicMock()
    view = memoryview(torch.zeros((num_chunks, row_bytes), dtype=torch.int8).numpy())
    mock.create_kv_memoryview.return_value = view
    return mock


# --- block placement -------------------------------------------------------


def test_place_blocks_skips_into_first_chunk():
    # 7 blocks from logical block 6, 4 blocks per chunk: logical 6..12.
    spec = GPULoadStoreSpec(list(range(100, 107)), group_sizes=[7], block_indices=[6])
    placement = place_blocks(spec, blocks_per_chunk=4)
    assert placement.chunk_idx.tolist() == [0, 0, 1, 1, 1, 1, 2]
    assert placement.position.tolist() == [2, 3, 0, 1, 2, 3, 0]
    assert placement.group.tolist() == [0] * 7
    assert placement.num_chunks == 3


def test_place_blocks_starts_each_group_on_a_new_chunk():
    spec = GPULoadStoreSpec(
        [10, 11, 12, 20, 21], group_sizes=[3, 2], block_indices=[4, 1]
    )
    placement = place_blocks(spec, blocks_per_chunk=4)
    assert placement.chunk_idx.tolist() == [0, 0, 0, 1, 1]
    assert placement.position.tolist() == [0, 1, 2, 1, 2]
    assert placement.group.tolist() == [0, 0, 0, 1, 1]
    assert placement.num_chunks == 2


def test_sub_gpu_spec_keeps_offsets_into_the_chunk():
    spec = GPULoadStoreSpec(list(range(100, 107)), group_sizes=[7], block_indices=[6])
    placement = place_blocks(spec, blocks_per_chunk=4)
    middle = sub_gpu_spec(spec, placement, 1, 2)
    assert middle.block_ids.tolist() == [102, 103, 104, 105]
    assert list(middle.group_sizes) == [4]
    assert list(middle.block_indices) == [8]
    head = sub_gpu_spec(spec, placement, 0, 1)
    assert head.block_ids.tolist() == [100, 101]
    assert list(head.block_indices) == [6]
    assert place_blocks(head, 4).position.tolist() == [2, 3]


def test_sub_gpu_spec_leaves_unused_groups_empty():
    spec = GPULoadStoreSpec(
        [10, 11, 12, 20, 21], group_sizes=[3, 2], block_indices=[4, 1]
    )
    placement = place_blocks(spec, blocks_per_chunk=4)
    second = sub_gpu_spec(spec, placement, 1, 2)
    assert second.block_ids.tolist() == [20, 21]
    assert list(second.group_sizes) == [0, 2]
    assert list(second.block_indices) == [4, 1]


# --- composite worker -------------------------------------------------------


class FakeCpuWorker:
    def __init__(self):
        self.loads: list[tuple[int, CPULoadStoreSpec, GPULoadStoreSpec]] = []
        self.finished: list[TransferResult] = []
        self.waited: list[set[int]] = []

    def submit_store(self, job_id, src_spec, dst_spec):
        return True

    def submit_load(self, job_id, src_spec, dst_spec):
        self.loads.append((job_id, src_spec, dst_spec))
        return True

    def get_finished(self):
        out, self.finished = self.finished, []
        return out

    def wait(self, job_ids):
        self.waited.append(set(job_ids))

    def shutdown(self):
        pass


class FakeLoader:
    def __init__(self):
        self.loads: list[tuple[int, list[str], np.ndarray, BlockPlacement]] = []
        self.finished: list[TransferResult] = []
        self.waited: list[set[int]] = []

    def submit(self, job_id, names, block_ids, placement):
        self.loads.append((job_id, list(names), block_ids, placement))
        return True

    def get_finished(self):
        out, self.finished = self.finished, []
        return out

    def wait(self, job_ids):
        self.waited.append(set(job_ids))

    def shutdown(self):
        pass


def _mixed_load():
    src = TieredLoadSpec([CPULoadStoreSpec([10]), DirectLoadSpec(1, ["a", "b"])])
    dst = GPULoadStoreSpec(list(range(100, 106)), group_sizes=[6], block_indices=[0])
    return src, dst


def test_worker_splits_a_mixed_load():
    cpu, loader = FakeCpuWorker(), FakeLoader()
    worker = TieringOffloadingWorker(cpu, {1: loader}, blocks_per_chunk=2)
    src, dst = _mixed_load()
    assert worker.submit_load(7, src, dst)

    (cpu_id, cpu_src, cpu_dst) = cpu.loads[0]
    assert cpu_id < 0
    assert cpu_src.chunk_ids.tolist() == [10]
    assert cpu_dst.block_ids.tolist() == [100, 101]
    (direct_id, names, block_ids, placement) = loader.loads[0]
    assert direct_id < 0 and direct_id != cpu_id
    assert names == ["a", "b"]
    assert block_ids.tolist() == [102, 103, 104, 105]
    assert placement.chunk_idx.tolist() == [0, 0, 1, 1]
    assert placement.position.tolist() == [0, 1, 0, 1]

    cpu.finished.append(TransferResult(cpu_id, True, 200, 0.5))
    assert worker.get_finished() == []
    loader.finished.append(TransferResult(direct_id, True, 400, 0.25))
    assert worker.get_finished() == [TransferResult(7, True, 600, 0.5)]


def test_worker_fails_the_load_when_a_segment_fails():
    cpu, loader = FakeCpuWorker(), FakeLoader()
    worker = TieringOffloadingWorker(cpu, {1: loader}, blocks_per_chunk=2)
    src, dst = _mixed_load()
    worker.submit_load(7, src, dst)
    cpu.finished.append(TransferResult(cpu.loads[0][0], True))
    loader.finished.append(TransferResult(loader.loads[0][0], False))
    (result,) = worker.get_finished()
    assert result.job_id == 7 and not result.success


def test_worker_passes_plain_loads_and_stores_through():
    cpu, loader = FakeCpuWorker(), FakeLoader()
    worker = TieringOffloadingWorker(cpu, {1: loader}, blocks_per_chunk=2)
    dst = GPULoadStoreSpec([5, 6], group_sizes=[2], block_indices=[0])
    worker.submit_load(3, CPULoadStoreSpec([1]), dst)
    assert cpu.loads[0][0] == 3
    cpu.finished.append(TransferResult(3, True))
    assert worker.get_finished() == [TransferResult(3, True)]
    assert not loader.loads


def test_worker_refuses_a_load_whose_chunks_do_not_match():
    worker = TieringOffloadingWorker(
        FakeCpuWorker(), {1: FakeLoader()}, blocks_per_chunk=2
    )
    src = TieredLoadSpec([DirectLoadSpec(1, ["a"])])
    dst = GPULoadStoreSpec([1, 2, 3], group_sizes=[3], block_indices=[0])
    assert not worker.submit_load(1, src, dst)


def test_worker_waits_on_the_segments_of_a_load():
    cpu, loader = FakeCpuWorker(), FakeLoader()
    worker = TieringOffloadingWorker(cpu, {1: loader}, blocks_per_chunk=2)
    src, dst = _mixed_load()
    worker.submit_load(7, src, dst)
    worker.wait({7})
    assert cpu.waited == [{cpu.loads[0][0]}]
    assert loader.waited == [{loader.loads[0][0]}]


# --- object store read plan -------------------------------------------------


def _loader(pages, chunk_offsets, bases, region_starts=None, group_sizes=None):
    loader = object.__new__(ObjDirectLoader)
    loader._pages = np.array(pages, dtype=np.int64)
    loader._chunk_offsets = np.array(chunk_offsets, dtype=np.int64)
    loader._bases = np.array(bases, dtype=np.int64)
    loader._region_starts = np.array(
        region_starts if region_starts is not None else sorted(bases), dtype=np.int64
    )
    sizes = group_sizes if group_sizes is not None else pages
    loader._group_copies = [list(enumerate(sizes))]
    return loader


def _placement(chunks, positions):
    return BlockPlacement(
        chunk_idx=np.array(chunks),
        position=np.array(positions),
        group=np.zeros(len(chunks), dtype=np.int64),
        num_chunks=max(chunks) + 1,
    )


def _reads(plan):
    return [tuple(int(x) for x in row) for row in zip(*plan)]


def test_plan_merges_a_chunk_into_consecutive_blocks():
    loader = _loader(pages=[100], chunk_offsets=[0], bases=[1 << 20])
    plan = loader._plan(np.array([10, 11, 12, 13]), _placement([0] * 4, [0, 1, 2, 3]))
    assert _reads(plan) == [(0, 0, (1 << 20) + 1000, 400)]


def test_plan_splits_scattered_blocks():
    loader = _loader(pages=[100], chunk_offsets=[0], bases=[1 << 20])
    plan = loader._plan(np.array([10, 12]), _placement([0, 0], [0, 1]))
    assert _reads(plan) == [
        (0, 0, (1 << 20) + 1000, 100),
        (0, 100, (1 << 20) + 1200, 100),
    ]


def test_plan_reads_each_tensor_from_its_part_of_the_chunk():
    # Two tensors, 4 blocks per chunk: tensor 1 starts at 4 * 100 in a chunk.
    loader = _loader(pages=[100, 50], chunk_offsets=[0, 400], bases=[1 << 20, 1 << 24])
    plan = loader._plan(np.array([5]), _placement([2], [1]))
    assert _reads(plan) == [
        (2, 100, (1 << 20) + 500, 100),
        (2, 450, (1 << 24) + 250, 50),
    ]


def test_plan_does_not_merge_across_registered_regions():
    base = 1 << 20
    loader = _loader(
        pages=[100], chunk_offsets=[0], bases=[base], region_starts=[base, base + 200]
    )
    plan = loader._plan(np.array([0, 1, 2, 3]), _placement([0] * 4, [0, 1, 2, 3]))
    assert _reads(plan) == [(0, 0, base, 200), (0, 200, base + 200, 200)]


def test_plan_does_not_merge_padded_pages():
    # A 90-byte page in a 100-byte slot leaves a gap after each page.
    loader = _loader(pages=[100], chunk_offsets=[0], bases=[1 << 20], group_sizes=[90])
    plan = loader._plan(np.array([3, 4]), _placement([0, 0], [0, 1]))
    assert len(_reads(plan)) == 2


# --- scheduler side ---------------------------------------------------------


class DirectTier(ExampleSecondaryTierManager):
    gpu_direct_load = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted_loads: list = []

    def submit_load(self, job_metadata):
        self.submitted_loads.append(job_metadata)
        super().submit_load(job_metadata)

    def direct_load_names(self, keys: Collection[OffloadKey]) -> list[str]:
        return [f"obj/{key.hex()}" for key in keys]


class TestDirectLoadManager:
    @pytest.fixture(autouse=True)
    def setup(self):
        region = _mock_mmap_region(5)
        self.primary = CPUPrimaryTierOffloadingManager(num_chunks=5, mmap_region=region)
        self.tier = DirectTier(
            offloading_spec=MagicMock(),
            primary_kv_view=region.create_kv_memoryview(),
            tier_type="direct",
        )
        self.manager = TieringOffloadingManager(
            primary_tier=self.primary, secondary_tiers=[self.tier]
        )
        self.ctx = ReqContext(req_id="r")
        self.manager.on_new_request(self.ctx)

    def _end_step(self):
        self.manager.on_schedule_end(
            ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
        )

    def _store_in_primary(self, keys):
        self.manager.prepare_store(keys, self.ctx)
        self.manager.complete_store(keys, self.ctx, success=True)

    def test_hit_needs_no_promotion(self):
        keys = to_keys(range(2))
        for key in keys:
            self.tier.chunks[key] = True
        assert [self.manager.lookup(k, self.ctx) for k in keys] == [
            LookupResult.HIT
        ] * 2
        self._end_step()
        self._end_step()
        assert self.tier.submitted_loads == []

        spec = self.manager.prepare_load(keys, self.ctx)
        assert isinstance(spec, TieredLoadSpec)
        (segment,) = spec.segments
        assert isinstance(segment, DirectLoadSpec)
        assert segment.tier_idx == 0
        assert segment.names == [f"obj/{k.hex()}" for k in keys]
        # Nothing was pinned in the primary tier, so nothing to release.
        self.manager.complete_load(keys, self.ctx)

    def test_mixed_load_has_one_segment_per_source(self):
        cpu_keys, direct_keys = to_keys(range(2)), to_keys(range(2, 4))
        self._store_in_primary(cpu_keys)
        for key in direct_keys:
            self.tier.chunks[key] = True
        keys = cpu_keys + direct_keys
        assert all(self.manager.lookup(k, self.ctx) is LookupResult.HIT for k in keys)
        # The cascade to the secondary tier may still pin the stored chunks.
        before = [self.primary._policy.get(key).ref_cnt for key in cpu_keys]

        spec = self.manager.prepare_load(keys, self.ctx)
        assert isinstance(spec, TieredLoadSpec)
        cpu_segment, direct_segment = spec.segments
        assert isinstance(cpu_segment, CPULoadStoreSpec)
        assert len(cpu_segment.chunk_ids) == 2
        assert isinstance(direct_segment, DirectLoadSpec)
        assert len(direct_segment.names) == 2
        pinned = [self.primary._policy.get(key).ref_cnt for key in cpu_keys]
        assert pinned == [n + 1 for n in before]

        self.manager.complete_load(keys, self.ctx)
        assert [self.primary._policy.get(key).ref_cnt for key in cpu_keys] == before

    def test_primary_hit_wins_over_an_earlier_direct_hit(self):
        keys = to_keys(range(1))
        self.tier.chunks[keys[0]] = True
        assert self.manager.lookup(keys[0], self.ctx) is LookupResult.HIT
        self._store_in_primary(keys)
        assert self.manager.lookup(keys[0], self.ctx) is LookupResult.HIT
        spec = self.manager.prepare_load(keys, self.ctx)
        assert isinstance(spec, CPULoadStoreSpec)
        self.manager.complete_load(keys, self.ctx)

    def test_requests_served_for_another_tier_still_promote(self):
        key = to_keys(range(1))[0]
        self.tier.chunks[key] = True
        result = self.manager.lookup(key, self.ctx, exclude_tier_idx=5)
        assert result is LookupResult.HIT_PENDING
        self._end_step()
        assert len(self.tier.submitted_loads) == 1


# --- staged loads -----------------------------------------------------------


class FakeObjAgent:
    """Stands in for a NIXL agent with an OBJ backend: a READ copies object
    bytes into local memory at the descriptor's address."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.names: dict[int, str] = {}
        self.reads: list[int] = []

    def register_memory(self, descs, mem_type):
        assert mem_type == "OBJ"
        for _, _, dev, name in descs:
            self.names[dev] = name
        return [dev for _, _, dev, _ in descs]

    def deregister_memory(self, reg):
        for dev in reg:
            del self.names[dev]

    def get_xfer_descs(self, descs, mem_type):
        return descs

    def initialize_xfer(self, op, local, remote, agent_name):
        assert op == "READ"
        return (local, remote)

    def transfer(self, handle):
        local, remote = handle
        for (addr, length, _), (offset, rlen, dev) in zip(local, remote):
            assert length == rlen
            data = self.objects[self.names[int(dev)]][offset : offset + length]
            ctypes.memmove(int(addr), data, int(length))
        self.reads.append(len(local))
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        pass


def _staged_loader(kv, objects, blocks_per_chunk, wave, chunk_bytes):
    loader = object.__new__(ObjStagedLoader)
    loader._layout = _KvLayout.build(kv, blocks_per_chunk)
    loader._chunk_bytes = chunk_bytes
    loader._worker_offset = 0
    loader._slot_bytes = loader._layout.slice_bytes + 64
    loader._wave = wave
    loader._staging = torch.zeros((2 * wave, loader._slot_bytes), dtype=torch.int8)
    loader._staging_base = loader._staging.data_ptr()
    loader._device = 0
    loader._name = "fake"
    loader._next_obj_dev_id = 1
    loader._agent = FakeObjAgent(objects)

    def copy(src, dst, size, stream):
        for s, d, n in zip(src, dst, size):
            ctypes.memmove(int(d), int(s), int(n))

    loader._copy = copy
    return loader


@pytest.mark.parametrize("wave", [1, 2, 8])
def test_staged_load_places_every_page(wave):
    bpc, nb = 4, 32
    pages = [64, 32]
    tensors = [torch.zeros((nb, p), dtype=torch.int8) for p in pages]
    kv = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=t, page_size_bytes=p)
            for t, p in zip(tensors, pages)
        ],
        group_data_refs=[
            [
                CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=p)
                for i, p in enumerate(pages)
            ]
        ],
    )
    chunk_bytes = sum(pages) * bpc + 128
    rng = np.random.default_rng(3)
    names = [f"obj/{i}" for i in range(3)]
    objects = {
        n: rng.integers(0, 256, chunk_bytes, dtype=np.uint8).tobytes() for n in names
    }
    # 10 blocks from logical block 2: chunks 0, 1, 2, scattered GPU ids
    block_ids = np.array([5, 6, 20, 21, 22, 23, 9, 1, 30, 31], dtype=np.int32)
    dst = GPULoadStoreSpec(block_ids.tolist(), group_sizes=[10], block_indices=[2])
    placement = place_blocks(dst, bpc)

    loader = _staged_loader(kv, objects, bpc, wave, chunk_bytes)
    nbytes = loader._load(_StagedJob(1, names, block_ids, placement), stream=None)

    assert nbytes == 3 * sum(pages) * bpc
    assert len(loader._agent.reads) == -(-3 // wave)
    offsets = [0, pages[0] * bpc]
    for b, k, j in zip(block_ids, placement.chunk_idx, placement.position):
        for t, page in enumerate(pages):
            start = offsets[t] + int(j) * page
            want = np.frombuffer(objects[names[k]][start : start + page], dtype=np.int8)
            assert np.array_equal(tensors[t][int(b)].numpy(), want)
    untouched = set(range(nb)) - set(block_ids.tolist())
    for t in range(len(pages)):
        assert not tensors[t][sorted(untouched)].any()


def test_staged_copy_plan_merges_consecutive_blocks():
    loader = object.__new__(ObjStagedLoader)
    loader._staging_base = 1 << 20
    loader._slot_bytes = 4096
    slot_of_chunk = np.array([1, 0])
    # chunk 0 (slot 1): two pages into consecutive blocks, then one apart
    src, dst, size = loader._copy_plan(
        chunk=np.array([0, 0, 0]),
        src_off=np.array([0, 100, 200]),
        dst=np.array([5000, 5100, 9000]),
        size=np.array([100, 100, 100]),
        slot_of_chunk=slot_of_chunk,
    )
    base = (1 << 20) + 4096
    assert src.tolist() == [base, base + 200]
    assert dst.tolist() == [5000, 9000]
    assert size.tolist() == [200, 100]
