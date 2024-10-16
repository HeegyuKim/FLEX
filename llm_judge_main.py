import fire
import os
import json, jsonlines
from utils.db_utils import load_database
from tqdm.auto import tqdm
from utils.llm_judge import OpenAIJudge
from utils.eval_utils import estimate_skip_length
from utils.dataset_schema import get_schema_text_dict
from utils import prompts
from datasets import load_dataset
import pandas as pd
import concurrent.futures
import threading
from pprint import pprint
from collections import defaultdict


def get_execute_result(db, db_id, query, max_rows=100):
    df = db.execute(db_id, query, to_pandas=True)
    df = df.drop_duplicates()

    if len(df) > max_rows:
        df_truncated = df.iloc[:max_rows]
        df_output = df_truncated.to_csv(index=False) + "... (truncated)\n"
    else:
        df_output = df.to_csv(index=False)
        
    return df_output + f"\nshape={df.shape}"

def calculate_ex(df1, df2):
    df1 = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
    df2 = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

    return df1.shape == df2.shape and (df1.values == df2.values).all()


def main(
    input_file: str,
    dataset: str,
    judge_model: str = "gpt-4o-mini",
    db_engine: str = "sqlite",
    batch_size: int = 1,
    num_threads: int = 1,
    revision: str = None,
    limit: int = None,
):
    if dataset == "bird-dev-mini":
        db_path = "./bird_mini_dev/llm/mini_dev_data/minidev/MINIDEV/dev_databases"
        questions = json.load(open("./bird_mini_dev/llm/mini_dev_data/minidev/MINIDEV/dev.json"))
        questions = {item["question_id"]: item for item in questions}

    elif dataset == "bird" or dataset == "bird-dev":
        db_path = "./bird_dev/llm/data/dev_databases"
        questions = json.load(open("./bird_dev/llm/data/dev.json"))
        questions = {item["question_id"]: item for item in questions}

    elif dataset == "spider":
        db_path = "./spider/database"
        questions = json.load(open("./spider/dev.json"))
        questions = {i: item for i, item in enumerate(questions)}

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    schema_dict = get_schema_text_dict(dataset, db_path=db_path, add_description=True)
            
    total_count = len(questions)

    items = json.load(open(input_file))
    if revision:
        output_file = input_file.replace(".json", f"_judge_{revision}.jsonl")
    else:
        output_file = input_file.replace(".json", "_judge.jsonl")

    db = load_database(db_engine, db_path=db_path)
    judge = OpenAIJudge(judge_model)

    skip_length = estimate_skip_length(output_file)
    if skip_length > 0:
        print(f"Skip {skip_length} items")

    def process_item(item):
        question_id = item["sql_idx"] if "sql_idx" in item else item["index"]
        question = questions[question_id]["question"]
        db_id = item["db"] if "db" in item else item["db_path"].split("/")[-1].replace(".sqlite", "")
        pred_sql = item.get("pred") or item["predicted_sql"]
        gt_sql = item.get("gold") or item["ground_truth"]

        if dataset == "spider":
            hardness = item["hardness"]
            try:
                pred_result = get_execute_result(db, db_id, pred_sql)
            except Exception as e:
                item["error"] = str(e)
                print(f"Error: {question_id}", e)
                ex = 0

            gt_result = get_execute_result(db, db_id, gt_sql)
            ex = item["ex"]
            strict = ex == 1 or ("order by" in gt_sql.lower() or "order by" in pred_sql.lower())

        elif dataset == "bird" or dataset == "bird-dev":
            hardness = questions[question_id]["difficulty"]
            ex = item["res"]
            strict = ex == 1
            pred_result, _ = prompts.execution_result2text_set(item.get("predicted_res"), deduplicate=strict)
            gt_result, _ = prompts.execution_result2text_set(item.get("ground_truth_res"), deduplicate=strict)

        if "error" not in item:
            judge_text, judge_result = judge.judge(
                schema_dict[db_id],
                question,
                pred_sql,
                pred_result,
                gt_sql,
                gt_result,
                hint=item.get("evidence"),
                verbose=True,
                strict=strict,
            )
        else:
            pred_result, gt_result = None, None
            judge_text, judge_result = item['error'], "error"

        return {
            "index": question_id,
            "schema": schema_dict[db_id],
            "question": question,
            "hint": item.get("evidence"),
            "pred_sql": pred_sql,
            "pred_exec": pred_result,
            "gt_sql": gt_sql,
            "gt_exec": gt_result,
            "judge_text": judge_text,
            "judge_result": judge_result,
            "hardness": hardness,
            "ex": ex
        }
    
    fout = jsonlines.open(output_file, "a")
    lock = threading.Lock()

    def process_batch(batch):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_item, item) for item in batch]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        results = [result for result in results if result is not None]
        results = sorted(results, key=lambda x: x['index'])
        with lock:
            for item in results:
                fout.write(item)

    if skip_length < total_count:
        batch = []
        for i, item in enumerate(tqdm(items)):
            if i < skip_length:
                continue
            if limit and i >= limit + skip_length:
                break
            batch.append(item)
            if len(batch) == batch_size:
                process_batch(batch)
                batch = []
        if batch:
            process_batch(batch)

    fout.close()

    df = pd.read_json(output_file, lines=True)
    # accuracy
    print("EX:", df.ex.mean())
    print("FLEX:", df.judge_result.value_counts(normalize=True))

    false_negative = df[(df["ex"] == 0) & (df["judge_result"] == True)]
    false_positive = df[(df["ex"] == 1) & (df["judge_result"] != True)]
    print("False Positive:", round(len(false_positive) / len(df) * 100, 2), len(false_positive))
    print("False Negative:", round(len(false_negative) / len(df) * 100, 2), len(false_negative))

    
if __name__ == "__main__":
    fire.Fire(main)