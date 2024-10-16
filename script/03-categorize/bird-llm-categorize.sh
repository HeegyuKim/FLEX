# python llm_judge_categorize.py "text2sql-baselines/bird/*_ex_judge.jsonl" bird-dev gpt-4o-2024-08-06 --batch_size 32 --num_threads 8

bird_cat() {
    model=$1
    python llm_judge_categorize.py "text2sql-baselines/bird/${model}_ex_judge.jsonl" bird-dev gpt-4o-2024-08-06 --batch_size 32 --num_threads 8
}

# BIRD_TOP10=[
#  "SuperSQL"
#  "CHESS-GPT-4o-mini"
#  "TA-ACL"
#  'DAIL_SQL_9-SHOT_MP',
#  'DAIL_SQL_9-SHOT_QM',
#  'DTS-SQL-BIRD-GPT4o-0823',
#  'SFT_CodeS_15B_EK',
#  'SFT_CodeS_7B_EK',
#  'SFT_CodeS_3B_EK',
#  'DAIL_SQL'
# ]
bird_cat "SuperSQL"
bird_cat "CHESS-GPT-4o-mini"
bird_cat "TA-ACL"
bird_cat "DAIL_SQL_9-SHOT_MP"
bird_cat "DAIL_SQL_9-SHOT_QM"
bird_cat "DTS-SQL-BIRD-GPT4o-0823"
bird_cat "SFT_CodeS_15B_EK"
bird_cat "SFT_CodeS_7B_EK"
bird_cat "SFT_CodeS_3B_EK"
bird_cat "DAIL_SQL"
