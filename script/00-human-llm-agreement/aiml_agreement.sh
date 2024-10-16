export AIML_API_KEY=YOUR_API_KEY
eval() {
    model="aiml/$1"
    # output_name=$2
 
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fn.jsonl $model
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fp.jsonl $model
}

eval "codellama/CodeLlama-70b-Instruct-hf"
eval "codellama/CodeLlama-34b-Instruct-hf"
eval "deepseek-ai/deepseek-coder-33b-instruct"