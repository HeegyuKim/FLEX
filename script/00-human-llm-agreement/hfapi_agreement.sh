
eval() {
    model=$1

    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fn.jsonl $model
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fp.jsonl $model
}

eval "hf-inference-api/meta-llama/Meta-Llama-3.1-70B-Instruct"
eval "hf-inference-api/mistralai/Mixtral-8x7B-Instruct-v0.1"

eval "hf-inference-api/meta-llama/Meta-Llama-3.1-405B-Instruct"
eval "hf-inference-api/meta-llama/Meta-Llama-3.1-70B-Instruct"
eval "hf-inference-api/meta-llama/Meta-Llama-3.1-8B-Instruct"

eval "hf-inference-api/Qwen/Qwen2.5-72B-Instruct"
eval "hf-inference-api/meta-llama/Meta-Llama-3-70B-Instruct"
eval "hf-inference-api/meta-llama/Meta-Llama-3-8B-Instruct"

eval "hf-inference-api/01-ai/Yi-Coder-9B-Chat" "Yi-Coder-9B-Chat"
eval "hf-inference-api/microsoft/Phi-3-medium-128k-instruct" "microsoft-Phi-3-medium-128k-instruct"

# eval "hf-inference-api/mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai-Mixtral-8x7B-Instruct-v0.1"
