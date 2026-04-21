# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the scheduler-side KV-connector offload gate that blocks
connector participation for requests that lack a ``prompt_cache_key`` when
``KVTransferConfig.require_cache_key_for_offload`` is enabled."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler

pytestmark = pytest.mark.cpu_test


def _gate(require_key: bool, prompt_cache_key: str | None) -> bool:
    """Drive _connector_offload_gated directly, with just enough state
    faked to exercise the flag+key decision without standing up a real
    scheduler / engine."""
    fake_scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                require_cache_key_for_offload=require_key,
            )
        )
    )
    fake_request = SimpleNamespace(prompt_cache_key=prompt_cache_key)
    return Scheduler._connector_offload_gated(fake_scheduler, fake_request)


def test_gate_off_never_blocks():
    assert _gate(require_key=False, prompt_cache_key=None) is False
    assert _gate(require_key=False, prompt_cache_key="anything") is False


def test_gate_on_blocks_unkeyed_requests():
    assert _gate(require_key=True, prompt_cache_key=None) is True
    assert _gate(require_key=True, prompt_cache_key="") is True


def test_gate_on_passes_keyed_requests():
    assert _gate(require_key=True, prompt_cache_key="tenant-A/thread-1") is False


def test_gate_no_kv_transfer_config_never_blocks():
    fake_scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=None)
    )
    fake_request = SimpleNamespace(prompt_cache_key=None)
    assert Scheduler._connector_offload_gated(fake_scheduler, fake_request) is False
