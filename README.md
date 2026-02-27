# Convolutional Context Propagation (CCP)

CCP uses a convolution-like pattern for LLM reasoning over large context: each chunk is converted into a local evidence node, then sliding windows fuse neighboring nodes across layers (`window`, `stride`) to produce progressively more abstract and global answers.

CCP is a layered long-context reasoning pipeline that:
- chunks long context,
- extracts local evidence per chunk,
- fuses evidence across layers in convolution-style windows,
- verifies the final answer for support.

This repository is CCP-only (no RLM comparison code).

## Method

CCP (Convolutional Context Propagation) processes long context in four stages:

1. **Plan**
   - A planner predicts task guidance from the query (for example: task type, output format, abstention policy).
   - This guidance conditions all later steps.

2. **Chunk + Extract**
   - The context is split into overlapping chunks (`--chunk-chars`, `--chunk-overlap`).
   - For each chunk, an extractor returns:
     - a local candidate fragment,
     - key points/evidence spans,
     - a relevance/confidence signal.
   - CCP keeps higher-signal chunk nodes and discards empty/weak ones.

3. **Convolution-Style Fusion Across Layers**
   - Nodes are fused in sliding windows of size `--window` with movement `--stride`.
   - This is repeated for `--layers` rounds, progressively abstracting and consolidating information.
   - Each fusion step outputs a new node (draft answer + fused points + confidence), similar to hierarchical feature aggregation.

4. **Verification**
   - A verifier checks whether the selected draft answer is valid/supported.
   - If validation fails, it emits a corrected answer.

### Convolution + Abstraction Graph

```mermaid
flowchart TB
  Q["Query"] --> P["Plan: task type, output format, abstention"]
  C["Long Context"] --> S["Chunk + Extract"]
  P --> S

  subgraph L0["Layer 0 (chunk nodes)"]
    N1["n1"] --- N2["n2"] --- N3["n3"] --- N4["n4"] --- N5["n5"]
  end

  S --> L0

  subgraph L1["Layer 1 (window fuse; window=w, stride=s)"]
    F1["f1 = fuse(n1..nw)"]
    F2["f2 = fuse(n1+s..nw+s)"]
    F3["f3 = ..."]
  end

  N1 --> F1
  N2 --> F1
  N3 --> F1
  N2 --> F2
  N3 --> F2
  N4 --> F2
  N5 --> F3

  subgraph L2["Layer 2 (higher abstraction)"]
    G1["g1 = fuse(f1..fk)"]
    G2["g2 = ..."]
  end

  F1 --> G1
  F2 --> G1
  F3 --> G2

  TOP["Top Node (best draft answer + fused evidence)"]
  V["Verify / Correct"]
  OUT["Final Answer"]

  G1 --> TOP
  G2 --> TOP
  TOP --> V --> OUT
```

### Why this helps on long context

- It avoids forcing one prompt to attend to the full document at once.
- It localizes evidence first, then propagates only compressed high-value signals upward.
- It gives explicit control over compute/quality trade-offs:
  - larger `window` or more `layers` => more aggregation capacity and cost,
  - smaller values => faster, cheaper inference.

### Signature Modes

- `adaptive` (default): task-aware QA signatures for general long-context benchmarks.
- `legacy`: memo-style/risk-style signatures for backward compatibility.

### What The DSPy Signatures Contain

CCP is implemented as four DSPy predictors. Each stage has a fixed input/output signature.

`adaptive` signatures:

```python
plan = dspy.Predict(
    "query -> task_type, expected_output_format, scoring_focus, abstention_policy"
)

extract = dspy.Predict(
    "chunk, query, task_type, expected_output_format, scoring_focus, abstention_policy "
    "-> relevance_0_to_1, key_points, candidate_fragment, evidence_spans, unresolved"
)

fuse = dspy.Predict(
    "window_findings, query, task_type, expected_output_format, scoring_focus, abstention_policy "
    "-> fused_points, draft_answer, confidence_0_to_1, unresolved"
)

verify = dspy.Predict(
    "query, task_type, expected_output_format, draft_answer, fused_points, unresolved "
    "-> is_valid, corrected_answer, verifier_notes"
)
```

What these fields do:
- `task_type`: coarse task label such as open QA, summarization, retrieval, classification.
- `expected_output_format`: constrains answer shape (short answer, list, paragraph, etc.).
- `scoring_focus`: tells downstream stages what to prioritize, usually faithfulness first.
- `abstention_policy`: tells the model how to behave when support is weak.
- `relevance_0_to_1`: local chunk usefulness score.
- `key_points`: distilled facts extracted from a chunk.
- `candidate_fragment`: chunk-local answer fragment.
- `evidence_spans`: quoted or anchored support used for validation.
- `unresolved`: missing information or uncertainty to carry upward.
- `fused_points`: merged evidence from multiple nodes.
- `draft_answer`: current best answer after fusion.
- `confidence_0_to_1`: fused node confidence score.
- `is_valid`: verifier judgment on whether the draft answer is supported.
- `corrected_answer`: verifier fallback if the draft answer is weak or incorrect.
- `verifier_notes`: explanation of the verifier decision.

`legacy` signatures:

```python
plan = dspy.Predict(
    "query -> answer_type, extraction_focus, abstention_rule"
)

extract = dspy.Predict(
    "chunk, query, answer_type, extraction_focus, abstention_rule "
    "-> evidence_found, quoted_evidence, atomic_facts, local_answer, confidence, missing_info"
)

fuse = dspy.Predict(
    "window_findings, query, answer_type, extraction_focus, abstention_rule "
    "-> fused_answer, fused_evidence, confidence, unresolved_points"
)

verify = dspy.Predict(
    "query, candidate_answer, fused_evidence, unresolved_points "
    "-> is_supported, corrected_answer, verifier_notes"
)
```

Use `adaptive` for general long-context QA benchmarks. Use `legacy` only when you want the older memo-style extraction behavior.

## CCP vs DSPy RLM

This repository focuses on CCP, but CCP is designed to be compared against DSPy RLM under matched settings.

### Conceptual Difference

| Aspect | CCP | RLM |
|---|---|---|
| Aggregation pattern | Fixed-depth convolution-style fusion over windows | Recursive reasoning / expansion |
| Main control knobs | `layers`, `window`, `stride`, chunking | recursion depth/branching style |
| Cost behavior | More predictable from fixed layer/window schedule | Can vary more with recursive branching |
| Typical strength | Stable local-to-global evidence compression | Flexible recursive decomposition |

### Fair Comparison Protocol

Use the same:
- dataset split,
- base model,
- decoding settings (temperature, max tokens),
- average token budget per query,
- judge and scoring method.

Report:
- quality (task metric or LLM-judge),
- latency,
- tokens/query and cost/query,
- win/tie/loss by sample.

## DSPy Implementation Snippet

```python
import dspy
from ccp.module import CPP

lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)
dspy.configure(lm=lm)

ccp = CPP(
    layers=2,
    window=2,
    stride=2,
    signature_mode="adaptive",
    enable_trace=True,
)

pred = ccp(
    context=long_context_text,
    query=user_query,
    chunk_chars=4000,
    chunk_overlap=400,
)

print(pred.answer)
print(pred.ccp_trace_json)  # intermediate layer trace
```

### Example: Module Configuration

```python
from ccp.module import CPP

ccp = CPP(
    layers=3,               # number of abstraction rounds
    window=3,               # how many neighboring nodes each fuse step sees
    stride=1,               # how far the fusion window moves each step
    signature_mode="adaptive",
    enable_trace=True,      # emit intermediate layer outputs
    extract_num_threads=4,  # parallel chunk extraction
    extract_max_errors=10,  # stop if too many chunk calls fail
)

pred = ccp(
    context=long_context_text,
    query=user_query,
    chunk_chars=3000,       # chunk size before fusion
    chunk_overlap=300,      # overlap between adjacent chunks
)
```

Parameter effect:
- `layers`: more layers increase abstraction depth and cost.
- `window`: larger windows combine more local evidence per fusion step.
- `stride`: smaller stride increases overlap between fusion windows.
- `chunk_chars`: larger chunks preserve more local context, but cost more per extract call.
- `chunk_overlap`: helps preserve facts that cross chunk boundaries.

## Repository Layout

- `ccp/module.py`: `CPP` DSPy module.
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

Download and unpack LongBench:

```bash
mkdir -p data/benchmarks/longbench
curl -L -o data/benchmarks/longbench/longbench_data.zip \
  https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip -o data/benchmarks/longbench/longbench_data.zip -d data/benchmarks/longbench
```

Or with `wget`:

```bash
mkdir -p data/benchmarks/longbench
wget -O data/benchmarks/longbench/longbench_data.zip \
  https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip -o data/benchmarks/longbench/longbench_data.zip -d data/benchmarks/longbench
```

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
