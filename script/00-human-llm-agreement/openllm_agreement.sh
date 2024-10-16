# model="vllm/google/gemma-2-27b-it"
# output_name="google-gemma-2-27b-it"
eval() {
    model=$1
    output_name=$2
 
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fn.jsonl $model
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fp.jsonl $model
}

export CUDA_VISIBLE_DEVICES=0

eval "hf/mistralai/Mistral-7B-Instruct-v0.3" "mistralai-Mistral-7B-Instruct-v0.3"
eval "hf/mistralai/Mistral-7B-Instruct-v0.2" "mistralai-Mistral-7B-Instruct-v0.2"
eval "hf/meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama-Meta-Llama-3.1-8B-Instruct"

eval "hf/microsoft/Phi-3-medium-128k-instruct" "Phi-3-medium-128k-instruct"
eval "hf/01-ai/Yi-Coder-9B-Chat" "Yi-Coder-9B-Chat"

eval "hf/google/gemma-2-9b-it" "google-gemma-2-9b-it"

eval "hf/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" "DeepSeek-Coder-V2-Lite-Instruct"
eval "hf/deepseek-ai/DeepSeek-V2-Lite-Chat" "DeepSeek-V2-Lite-Chat"

eval "hf/deepseek-ai/deepseek-coder-7b-instruct-v1.5" "deepseek-coder-7b-instruct-v1.5"
eval "hf/deepseek-ai/deepseek-coder-6.7b-instruct" "deepseek-coder-6.7b-instruct"

eval "hf/Qwen/Qwen2.5-Coder-7B-Instruct"
eval "hf/Qwen/Qwen2.5-7B-Instruct"
