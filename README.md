# Convolutional Context Propagation (CCP)

CCP is a layered long-context reasoning pipeline that:
- chunks long context,
- extracts local evidence per chunk,
- fuses evidence across layers in convolution-style windows,
- verifies the final answer for support.

This repository is CCP-only (no RLM comparison code).

## Repository Layout

- `ccp/module.py`: `CCPPipeline` DSPy module.
- `ccp/signatures.py`: signature definitions (legacy/adaptive).
- `ccp/judge.py`: LLM-as-judge module.
- `ccp/io_utils.py`: JSONL and chunking utilities.
- `ccp/scoring.py`: scoring and evidence parsing helpers.
- `ccp/runner.py`: CLI runtime logic.
- `ccp/pipeline.py`: backward-compatible re-export layer.
- `scripts/run_ccp.py`: CLI wrapper for running CCP.
- `scripts/prepare_longbench.py`: converts raw LongBench task files into one eval JSONL.
- `configs/`: example command settings.
- `tests/`: lightweight non-API tests.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set API key in your shell:

```bash
export OPENAI_API_KEY=your_key_here
```

## Data Preparation

Place LongBench task files under:

`data/benchmarks/longbench/data/*.jsonl`

Then build a unified eval file:

```bash
python scripts/prepare_longbench.py \
  --longbench-dir data/benchmarks/longbench/data \
  --output-jsonl data/longbench_eval.jsonl \
  --max-total 500
```

## Run CCP

```bash
python scripts/run_ccp.py \
  --dataset-jsonl data/longbench_eval.jsonl \
  --limit 20 \
  --model openai/gpt-4o-mini \
  --ccp-signature adaptive \
  --progress-every 1 \
  --show-progress \
  --save-predictions artifacts/ccp_longbench_20.jsonl \
  --save-traces artifacts/ccp_traces
```

## Quick Sanity Test

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Notes

- `--ccp-signature legacy` is available for backward compatibility.
- `--cap-signature` is accepted as an alias to avoid breaking older scripts.
- LLM judge score is `0-10`; higher is better.
