
for file in text2sql-baselines/test-spider/*.sql; do
    echo "Processing $file"
    bash bird_dev/eval.sh $file
done