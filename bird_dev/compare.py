import json
import pandas as pd
from pprint import pprint

old_file = "bird_dev/experiment_data/dataset/dev.json"
new_file = "bird_dev/llm/data/dev.json"

def read_file(filename: str):
    with open(filename, "r") as f:
        items = json.load(f)
    items = sorted(items, key=lambda x: x["question_id"])
    return items

old_items = read_file(old_file)
new_items = read_file(new_file)

df = pd.DataFrame({
    "old_question": [item["question"].strip() for item in old_items],
    "new_question": [item["question"].strip() for item in new_items],
    "old_query": [item["SQL"].strip() for item in old_items],
    "new_query": [item["SQL"].strip() for item in new_items],
})

# check question difference
diff_question = df[df["old_question"] != df["new_question"]]
for i, row in diff_question.iterrows():
    print(f"Question ID: {i}")
    print(f"Old: {row['old_question']}")
    print(f"New: {row['new_question']}")
    print()

# check SQL difference 
diff_sql = df[df["old_query"] != df["new_query"]]
for i, row in diff_sql.iterrows():
    print(f"Question ID: {i}")
    print(f"Old: {row['old_query']}")
    print(f"New: {row['new_query']}")
    print()
    
print(f"Total different questions: {len(diff_question)}")
print(f"Total different SQLs: {len(diff_sql)}")

