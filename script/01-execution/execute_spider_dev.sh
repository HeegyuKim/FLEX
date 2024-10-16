
for file in text2sql-baselines/spider_dev/*.sql; do
    echo "Processing $file"
    python -m spider_eval.evaluation --pred $file
done