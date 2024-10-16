
eval() {
    model=$1

    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fn.jsonl $model
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fp.jsonl $model
}

export CUDA_VISIBLE_DEVICES=0
eval "prometheus"
export CUDA_VISIBLE_DEVICES=0,1,2,3
eval "prometheus-8x7b"