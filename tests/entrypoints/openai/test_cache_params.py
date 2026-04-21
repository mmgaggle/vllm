# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the shared prompt-cache request parameter helpers and for
the OpenAI protocol validators that accept ``prompt_cache_key`` and
``prompt_cache_retention``."""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.cache_params import (
    KV_TRANSFER_RETENTION_KEY,
    merge_cache_retention,
    parse_cache_retention,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("default", None),
        ("", None),
        ("30s", 30),
        ("15m", 15 * 60),
        ("24h", 24 * 3600),
        ("7d", 7 * 86400),
    ],
)
def test_parse_cache_retention_valid(raw, expected):
    assert parse_cache_retention(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["forever", "24", "24h24m", "-1h", "0h", "h", "1.5h", "24H"],
)
def test_parse_cache_retention_invalid(raw):
    with pytest.raises(ValueError):
        parse_cache_retention(raw)


def test_merge_cache_retention_adds_seconds():
    merged = merge_cache_retention({"foo": "bar"}, "24h")
    assert merged == {"foo": "bar", KV_TRANSFER_RETENTION_KEY: 86400}


def test_merge_cache_retention_creates_dict_when_missing():
    merged = merge_cache_retention(None, "15m")
    assert merged == {KV_TRANSFER_RETENTION_KEY: 900}


def test_merge_cache_retention_default_is_passthrough():
    assert merge_cache_retention(None, "default") is None
    original = {"foo": "bar"}
    merged = merge_cache_retention(original, None)
    # Unchanged identity when retention is None / 'default'.
    assert merged is original


def test_merge_cache_retention_does_not_mutate_input():
    original = {"foo": "bar"}
    merged = merge_cache_retention(original, "24h")
    assert original == {"foo": "bar"}
    assert merged is not original


def test_chat_request_accepts_prompt_cache_fields():
    req = ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "hi"}],
        prompt_cache_key="tenant-A/session-1",
        prompt_cache_retention="24h",
    )
    assert req.prompt_cache_key == "tenant-A/session-1"
    assert req.prompt_cache_retention == "24h"


def test_chat_request_rejects_empty_prompt_cache_key():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="dummy",
            messages=[{"role": "user", "content": "hi"}],
            prompt_cache_key="",
        )


def test_chat_request_rejects_bad_retention():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="dummy",
            messages=[{"role": "user", "content": "hi"}],
            prompt_cache_retention="forever",
        )


def test_completion_request_retention_reaches_sampling_params():
    req = CompletionRequest(
        model="dummy",
        prompt="hi",
        prompt_cache_retention="24h",
    )
    params = req.to_sampling_params(max_tokens=4, default_sampling_params={})
    extra = params.extra_args or {}
    assert extra["kv_transfer_params"][KV_TRANSFER_RETENTION_KEY] == 86400


def test_completion_request_retention_default_does_not_touch_kv_params():
    req = CompletionRequest(
        model="dummy",
        prompt="hi",
        prompt_cache_retention="default",
    )
    params = req.to_sampling_params(max_tokens=4, default_sampling_params={})
    extra = params.extra_args or {}
    assert "kv_transfer_params" not in extra


def test_completion_request_retention_preserves_user_kv_params():
    req = CompletionRequest(
        model="dummy",
        prompt="hi",
        kv_transfer_params={"foo": "bar"},
        prompt_cache_retention="15m",
    )
    params = req.to_sampling_params(max_tokens=4, default_sampling_params={})
    extra = params.extra_args or {}
    assert extra["kv_transfer_params"] == {
        "foo": "bar",
        KV_TRANSFER_RETENTION_KEY: 900,
    }


def test_prompt_cache_key_is_not_cache_salt():
    """They are distinct fields and do not alias onto each other."""
    req = CompletionRequest(
        model="dummy",
        prompt="hi",
        cache_salt="s" * 43,
        prompt_cache_key="shared-prefix",
    )
    assert req.cache_salt == "s" * 43
    assert req.prompt_cache_key == "shared-prefix"
