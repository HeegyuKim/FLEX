
eval() {
    file=$1
    echo $file
    python llm_judge_main.py "$file" "bird-dev" "gpt-4o-2024-08-06" --batch_size 32 --num_threads 4
}

# for file in text2sql-baselines/bird/*_ex.json; do
#     eval $file
# done
# eval text2sql-baselines/bird/SFT_CodeS_3B_EK_ex.json

eval text2sql-baselines/bird/SuperSQL_ex.json
# eval text2sql-baselines/bird/TA-ACL_ex.json
# eval text2sql-baselines/bird/SFT_CodeS_7B_EK_ex.json
# python llm_judge_spider_batch.py