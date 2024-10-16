predicted_sql_path=$1

# cd bird_dev/llm/
# db_root_path='./data/dev_databases/'
# diff_json_path='./data/dev.json'
# ground_truth_path='./data/'

db_root_path='bird_dev/llm/data/dev_databases/'
diff_json_path='bird_dev/llm/data/dev.json'
ground_truth_path='bird_dev/llm/data/'

# db_root_path='bird_dev/experiment_data/dataset/dev/dev_databases/'
# diff_json_path='bird_dev/experiment_data/dataset/dev/dev.json'
# ground_truth_path='bird_dev/experiment_data/dataset/dev/'

data_mode='dev'
num_cpus=16
meta_time_out=600.0
mode_gt='gt'
mode_predict='gpt'

echo '''starting to compare with knowledge for ex'''
python3 -u bird_dev/llm/src/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}
