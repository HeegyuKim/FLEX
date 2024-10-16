
eval() {
    model="cohere/$1"
    # output_name=$2
 
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fn.jsonl $model
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fp.jsonl $model
}

export COHERE_API_KEY=YOUR_API_KEY
eval "command-r-plus-08-2024"
eval "command-r-plus-04-2024"
eval "command-r-08-2024"
eval "command-r-03-2024"

