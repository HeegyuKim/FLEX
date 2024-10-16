from llm_judge_main import main
from glob import glob
import pandas as pd

skip_patterns = ["500"] + [str(i) for i in range(1000, 8000, 1000)]

input_files = glob("text2sql-baselines/spider_dev/*_ex.json")
rows = []
for file in input_files:
    # skip pattern check
    if any(skip_pattern in file for skip_pattern in skip_patterns):
        continue

    df = pd.read_json(file)
    ex = df.ex.mean()
    print(f"{file}: {ex}")
    rows.append({"file": file, "ex": ex})

    main(file, "spider", "gpt-4o-2024-08-06", batch_size=32, num_threads=8)#, limit=256)
    # main(file, "spider", "gpt-4o-mini")

df = pd.DataFrame(rows)
print(df.describe())