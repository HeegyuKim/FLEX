eval() {
    model="togetherai/$1"
 
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fn.jsonl $model
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fp.jsonl $model
}

export TOGETHER_API_KEY=YOUR_API_KEY
eval "Qwen/Qwen2-72B-Instruct"
eval "Qwen/Qwen1.5-110B-Chat"
eval "databricks/dbrx-instruct"
eval "mistralai/Mixtral-8x22B-Instruct-v0.1"
eval "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"