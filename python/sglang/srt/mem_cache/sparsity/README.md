# Sparse KV-cache runtime

This package contains the shared sparse-attention control plane used by
HiSparse and the DeepSeek-V4 host-backed decode paths. It selects a sparse
algorithm, adapts attention metadata, and coordinates the request lifecycle;
the physical device and host KV pools remain in the neighboring `mem_cache`
modules.

For deployment requirements and complete launch examples, see the
[HiSparse user guide](../../../../../docs/docs/advanced_features/hisparse_guide.mdx).

## Package layout

| Path | Responsibility |
| --- | --- |
| `runtime.py` | Parse activation settings and resolve whether the sparse runtime is enabled for the current server role. |
| `factory.py` | Parse `SparseConfig`, select an algorithm and backend adaptor, and construct the process-wide coordinator. |
| `core/` | Own request tracking and the sparse-attention lifecycle hooks. |
| `algorithms/` | Select important KV positions and maintain algorithm-specific representations. |
| `backend/` | Translate selected positions into metadata understood by an attention backend. |

The intended dependency direction is:

```text
scheduler / model runner
          |
          v
   SparseCoordinator
      /         \
     v           v
algorithm   backend adaptor
     \           /
      v         v
       KV-cache pools
```

## Runtime activation

There are two supported activation paths:

1. `--enable-hisparse` enables the legacy HiSparse path.
2. Explicitly setting `dsv4_prefetch_mode` in `--hisparse-config` enables the
   DeepSeek-V4 host-backed runtime without the legacy flag.

For DeepSeek-V4, `dsv4_prefetch_mode` accepts:

- `scout` (default when HiSparse is already enabled): keep resident hits on
  GPU and evaluate host misses on CPU.
- `infinigen`: recall the predicted working set from host on a side stream.

The deprecated aliases `cpu` and `h2d` map to `scout` and `infinigen` and
produce a warning. Merely omitting the option does **not** activate sparse
execution. On a PD prefill server, an explicit DeepSeek-V4 mode is ignored
unless the legacy flag is also enabled; the optimization belongs on the
decode side.

Example for a regular server or PD decode server:

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/deepseek-v4 \
  --disable-decode-cuda-graph \
  --hisparse-config='{"top_k":512,"device_buffer_size":4096,"dsv4_prefetch_mode":"scout"}'
```

`runtime.resolve_sparse_runtime_policy()` is the single source of truth for
activation decisions. Callers should consume the returned policy instead of
reimplementing checks for server arguments or disaggregation roles.

## Coordinator lifecycle

`SparseCoordinator` is created after the KV pools and is called at these
boundaries:

1. `on_request_begin` initializes per-request tracking.
2. `attention_end` builds or updates the algorithm representation during
   prefill.
3. `forward_begin`, `attention_begin`, `attention_end`, and `forward_end`
   coordinate retrieval and offload during decode.
4. `on_request_end` clears request-local state.

The scheduler must also release any host/device staging resources when a
request is aborted. Cleanup should be idempotent because aborts may race with
asynchronous transfer completion or arrive after a request has left a queue.

## Extending the package

To add a retrievable sparse algorithm:

1. Implement `BaseSparseAlgorithm` under `algorithms/`.
2. Add its constructor to `_ALGORITHM_REGISTRY` in `factory.py`.
3. Add or reuse a backend adaptor in `backend/`.
4. Keep algorithm-specific JSON fields in `SparseConfig.sparse_extra_config`;
   only promote settings used by the shared runtime to explicit fields.
5. Add focused unit tests under `test/registered/unit/mem_cache/` and config
   parsing tests under `test/registered/mem_cache/`.

Avoid reading `server_args` throughout the execution path. Parse once in the
factory/runtime layer, validate values early, and pass typed configuration or
the resolved runtime policy downstream.

## Tests

Useful focused checks include:

```bash
python3 -m pytest test/registered/mem_cache/test_hisparse_config.py
python3 -m pytest test/registered/unit/mem_cache/test_hisparse_allocator.py
python3 -m pytest test/registered/unit/mem_cache/test_dsv4_sparse_runtime_pool.py
python3 -m pytest test/registered/unit/managers/test_hisparse_unit.py
```

The DeepSeek-V4 end-to-end coverage requires compatible CUDA hardware and is
located in `test/registered/models_e2e/test_deepseek_v4_sparse_runtime_h200.py`.
