

eval() {
    model=$1
    name=$2

    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fn.jsonl $model
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fp.jsonl $model
}

eval "llm-sql-solver/gpt-4o-2024-08-06"
# eval "llm-sql-solver/MiniatureAndMull/gpt-4o-2024-08-06"
# eval "llm-sql-solver/ExplainAndCompare/gpt-4o-2024-08-06"