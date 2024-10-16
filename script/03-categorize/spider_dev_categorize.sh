# ['SuperSQL',
#  'DINSQL',
#  'DAILSQL_SC',
#  'DAILSQL',
#  'TA-ACL',
#  'SFT_CodeS_7B',
#  'SFT_CodeS_15B',
#  'C3_SQL',
#  'SFT_Deepseek_Coder_7B',
#  'SFT_CodeS_3B']


spider_cat() {
    model=$1
    echo "Categorizing ${model}"
    python llm_judge_categorize.py "text2sql-baselines/spider_dev/${model}_ex_judge.jsonl" spider gpt-4o-2024-08-06 --batch_size 32 --num_threads 8
}

spider_cat "SuperSQL"
spider_cat "DINSQL"
spider_cat "DAILSQL_SC"
spider_cat "DAILSQL"
spider_cat "TA-ACL"

spider_cat "SFT_CodeS_7B"
spider_cat "SFT_CodeS_15B"
spider_cat "C3_SQL"
spider_cat "SFT_Deepseek_Coder_7B"
spider_cat "SFT_CodeS_3B"

