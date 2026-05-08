# Repository Guidelines

## Project Structure & Module Organization

`nanovllm/` contains the Python package. Core public entry points live in `nanovllm/llm.py`, `nanovllm/sampling_params.py`, and `nanovllm/config.py`. Runtime internals are grouped by concern: `nanovllm/engine/` handles scheduling, block management, model running, and sequence state; `nanovllm/layers/` contains attention, linear, normalization, activation, embedding, rotary, and sampling layers; `nanovllm/models/` holds model definitions such as Qwen3; `nanovllm/utils/` contains loader and context helpers. Root-level scripts include `example.py` for a smoke run, `bench.py` for throughput checks, and `install_by_uv.sh` for editable environment setup. Static assets are in `assets/`.

## Build, Test, and Development Commands

- `bash install_by_uv.sh`: creates `.venv`, installs CUDA-compatible PyTorch when detected, and installs this package in editable mode.
- `source .venv/bin/activate`: activates the local development environment.
- `pip install -e .`: installs the package from `pyproject.toml` when managing the environment manually.
- `python example.py --model_path ~/huggingface/Qwen3-0.6B/`: runs a basic generation smoke test.
- `python bench.py`: runs the local benchmark using the default Qwen3 path in the script.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax and keep style consistent with the existing code: 4-space indentation, concise dataclasses where appropriate, lowercase module names, `snake_case` functions and variables, and `PascalCase` classes. Prefer clear tensor-shape names and explicit configuration fields over hidden globals. No formatter or linter is currently configured, so avoid broad mechanical rewrites and keep changes narrowly scoped.

## Testing Guidelines

There is no committed test suite yet. For behavior changes, add focused tests if you introduce a test framework, or document manual validation in the pull request. At minimum, run `python example.py --model_path <local-model-dir>` for API and generation changes. For scheduler, cache, attention, or batching changes, also run `python bench.py` or a smaller local benchmark that exercises varied prompt and output lengths.

## Commit & Pull Request Guidelines

Recent history uses short imperative or scoped messages, for example `fix cache hit`, `support chunked prefill and fix minor bug`, and `fix(model_runner): correct seqlen_k to chunk boundary in prepare_prefill`. Keep commits focused and describe the affected subsystem when useful. Pull requests should include a concise problem statement, implementation summary, validation commands and results, linked issues when applicable, and notes about model, CUDA, GPU, or memory assumptions.

## Security & Configuration Tips

Do not commit model weights, downloaded Hugging Face artifacts, virtual environments, or machine-specific paths. Keep large local assets outside the repository and pass paths through script arguments where available.
