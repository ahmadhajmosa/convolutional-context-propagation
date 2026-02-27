# Data

This repository does not commit benchmark datasets.

## LongBench

Expected raw path for converter input:

`data/benchmarks/longbench/data/*.jsonl`

Build unified eval file:

```bash
python scripts/prepare_longbench.py \
  --longbench-dir data/benchmarks/longbench/data \
  --output-jsonl data/longbench_eval.jsonl
```

