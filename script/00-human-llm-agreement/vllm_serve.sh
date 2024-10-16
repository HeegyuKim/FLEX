export VLLM_ATTENTION_BACKEND=FLASHINFER


eval() {
    model=$1

    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fn.jsonl $model
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fp.jsonl $model
}

eval "vllm/deepseek-ai/deepseek-coder-33b-instruct"
eval "vllm/google/gemma-2-27b-it"