.PHONY: setup test run-small

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

test:
	python -m unittest discover -s tests -p "test_*.py" -v

run-small:
	python scripts/run_ccp.py \
		--dataset-jsonl data/longbench_eval.jsonl \
		--limit 20 \
		--model openai/gpt-4o-mini \
		--ccp-signature adaptive \
		--save-predictions artifacts/ccp_longbench_20.jsonl

