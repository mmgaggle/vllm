# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for OpenAI-compatible prompt-cache request parameters.

These back the ``prompt_cache_key`` and ``prompt_cache_retention`` fields that
sit alongside ``cache_salt`` on the OpenAI-compatible request schemas.
"""

import re
from typing import Any

_RETENTION_RE = re.compile(r"^(?P<n>\d+)(?P<unit>[smhd])$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

KV_TRANSFER_RETENTION_KEY = "cache_retention_s"


def parse_cache_retention(value: str | None) -> int | None:
    """Parse an OpenAI-compatible ``prompt_cache_retention`` hint.

    Accepts ``None``, ``"default"`` (both yield ``None`` — interpret as
    engine/connector default), or ``<int><unit>`` where ``unit`` is one of
    ``s``, ``m``, ``h``, ``d``. Returns the retention in seconds or ``None``.
    Raises ``ValueError`` on any other input.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("prompt_cache_retention must be a string if provided.")
    value = value.strip()
    if not value or value == "default":
        return None
    match = _RETENTION_RE.match(value)
    if match is None:
        raise ValueError(
            "prompt_cache_retention must be 'default' or '<int><unit>' where "
            "unit is one of 's', 'm', 'h', 'd' (e.g. '24h')."
        )
    n = int(match.group("n"))
    if n == 0:
        raise ValueError("prompt_cache_retention must be positive.")
    return n * _UNIT_SECONDS[match.group("unit")]


def merge_cache_retention(
    kv_transfer_params: dict[str, Any] | None,
    prompt_cache_retention: str | None,
) -> dict[str, Any] | None:
    """Return ``kv_transfer_params`` augmented with a parsed retention hint.

    If ``prompt_cache_retention`` is set and parses to a non-None number of
    seconds, the result is a (possibly new) dict carrying
    ``{KV_TRANSFER_RETENTION_KEY: seconds}``. The input dict is not mutated.
    """
    seconds = parse_cache_retention(prompt_cache_retention)
    if seconds is None:
        return kv_transfer_params
    merged: dict[str, Any] = (
        dict(kv_transfer_params) if kv_transfer_params else {}
    )
    merged[KV_TRANSFER_RETENTION_KEY] = seconds
    return merged
