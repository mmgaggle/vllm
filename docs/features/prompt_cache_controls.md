# Prompt Cache Controls

vLLM accepts two per-request hints that mirror OpenAI's prompt-caching
controls:

- `prompt_cache_key` — a deterministic partition tag. Requests sharing the
  same key (and the same `cache_salt`) coalesce into the same cached prefix;
  requests with different keys have disjoint cache.
- `prompt_cache_retention` — a retention hint, either `"default"` or an
  integer followed by one of `s`, `m`, `h`, `d` (e.g. `"24h"`).

They sit on top of [Automatic Prefix Caching](automatic_prefix_caching.md)
and the KV-connector subsystem (see [NixlConnector
Usage](nixl_connector_usage.md) for a representative connector guide). They
are accepted on `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
and the pooling/embeddings endpoints.

## Semantics

`prompt_cache_key` and the existing `cache_salt` are **orthogonal**:

| Field                | Purpose                                                                 | Chosen by            |
| -------------------- | ----------------------------------------------------------------------- | -------------------- |
| `cache_salt`         | Privacy/tenant isolation — random, unpredictable, per end-user          | Server or client (random) |
| `prompt_cache_key`   | Coalescing tag — deliberately shared across requests that should hit    | Client (deterministic) |

Both are folded into the APC block-hash for block 0 (the block-hash chain
propagates the partition to every downstream block), but they live in
distinct namespaces so a `prompt_cache_key` that happens to equal a
`cache_salt` value does not alias onto it.

`prompt_cache_retention` is parsed into seconds (`"default"` → no hint),
stored on the request, and forwarded to KV connectors via
`request.kv_transfer_params["cache_retention_s"]`. It does **not** change APC
eviction inside GPU memory today — connectors that honor TTLs (e.g. remote
shared stores with a retention policy) pick it up from `kv_transfer_params`.

## Sending the fields from an OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "<long shared system prompt>"},
        {"role": "user", "content": "Hello"},
    ],
    extra_body={
        "prompt_cache_key": "tenant-A/session-1",
        "prompt_cache_retention": "24h",
    },
)
```

Two calls with the same messages and the same `prompt_cache_key` will share
the cached prefix; changing the key produces a fresh miss.

## Deployment pattern: VRAM APC for all, KV offload only for tagged requests

A common multi-tenant shape is:

- every request benefits from in-VRAM APC (cheap, stays on the box),
- only requests that are explicitly tagged with a `prompt_cache_key` cross
  the KV-connector boundary into shared/remote storage.

Turn that on with the opt-in `require_cache_key_for_offload` flag on
`KVTransferConfig`. It is `False` by default so existing deployments are
unchanged.

```bash
vllm serve <MODEL> \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "LLMDKVCacheConnector",
    "kv_connector_module_path": "llmd_kv_cache.vllm_connector",
    "kv_role": "kv_both",
    "require_cache_key_for_offload": true
  }'
```

With this configuration:

- An untagged request hits only GPU APC. The scheduler short-circuits both
  `get_num_new_matched_tokens()` and `request_finished()` for that request,
  so the connector never sees it — no remote lookup, no offload, no state to
  clean up.
- A request with a non-empty `prompt_cache_key` goes through the connector
  as usual. The connector sees `request.prompt_cache_key` directly and can
  read `request.kv_transfer_params["cache_retention_s"]` for TTL hints.

The gate is connector-agnostic: it lives in the vLLM scheduler, so it
applies to any in-tree connector and to third-party connectors loaded via
`kv_connector_module_path`, with no connector-side code changes.

### With `llm-d-kv-cache`

[`llm-d-kv-cache`](https://github.com/llm-d/llm-d-kv-cache) is a third-party
KV-cache connector loaded dynamically via `kv_connector_module_path`. The
plumbing above relies on two vLLM-side behaviors only — (1) hashing
`prompt_cache_key` into APC block hashes and (2) the scheduler-side
`require_cache_key_for_offload` gate — both of which work regardless of
which connector is in use.

Pieces that a connector like `llm-d-kv-cache` needs to implement
**connector-side** to take full advantage of these controls:

- **TTL enforcement.** The retention hint arrives as an integer in
  `request.kv_transfer_params["cache_retention_s"]`. A connector that wants
  to act on it needs to forward the TTL to its backing store at write time
  (e.g. Redis `EXPIRE`, object-store object lifecycle, or a custom
  bookkeeping layer). Connectors that ignore the field simply preserve
  today's behavior.
- **Per-tenant / per-key routing.** A connector that wants to steer
  requests to different backends or namespaces based on
  `prompt_cache_key` can read `request.prompt_cache_key` directly in its
  scheduler-side methods (`get_num_new_matched_tokens`, `request_finished`).
  vLLM does not prescribe a routing scheme; see the project's own README for
  which parts, if any, are implemented today.

Consult the `llm-d-kv-cache` documentation for the current state of
retention and key-based routing support in that connector. If neither is
implemented yet, the partitioning (axis 1) and the offload gate (axis 2)
still apply and still deliver the "VRAM APC for all, tagged-only egress"
property — the retention hint is simply inert on the connector side until
the connector adds support.

## What these controls do not change

- APC in-GPU eviction is still LRU; retention does not pin blocks or bias
  the free-block queue.
- Existing `cache_salt` semantics are unchanged. Clients that used it
  continue to work; vLLM treats it as a privacy-isolation salt, not as a
  coalescing tag.
- The gate has no effect when no KV connector is configured.

## Verifying it works

- Issue two identical requests with the same `prompt_cache_key` and confirm
  `vllm:gpu_prefix_cache_hit_rate` rises on the second call.
- Change the `prompt_cache_key`; the hit rate for the shared prefix should
  drop to zero.
- With a connector configured and `require_cache_key_for_offload=true`,
  confirm in your connector's logs/metrics that untagged requests do not
  produce any remote reads or writes.
