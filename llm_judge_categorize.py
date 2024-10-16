import fire
import os, glob
import json, jsonlines
from utils.db_utils import load_database
from tqdm.auto import tqdm
from utils.llm_judge import OpenAIJudge
from utils.eval_utils import estimate_skip_length
from utils.dataset_schema import get_schema_text_dict
import pandas as pd
import concurrent.futures
import threading
from pprint import pprint


pd.set_option('future.no_silent_downcasting', True)

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
    if num_threads > 1 and batch_size == 1:
        batch_size = num_threads * 2    

    if dataset == "bird-dev-mini":
        db_path = "./bird_mini_dev/llm/mini_dev_data/minidev/MINIDEV/dev_databases"
        questions = json.load(open("./bird_mini_dev/llm/mini_dev_data/minidev/MINIDEV/dev.json"))
        questions = {item["question_id"]: item for item in questions}
        db_ids = [item["db_id"] for item in questions.values()]

    elif dataset == "bird" or dataset == "bird-dev":
        db_path = "./bird_dev/llm/data/dev_databases"
        questions = json.load(open("./bird_dev/llm/data/dev.json"))
        questions = {item["question_id"]: item for item in questions}
        db_ids = [item["db_id"] for item in questions.values()]

    elif dataset == "spider":
        db_path = "./spider/database"
        questions = json.load(open("./spider/dev.json"))
        questions = {i: item for i, item in enumerate(questions)}
        db_ids = [item["db_id"] for item in questions.values()]

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    schema_dict = get_schema_text_dict(dataset, db_path=db_path, add_description=True)
            
    total_count = len(questions)

    items = list(jsonlines.open(input_file))
    if revision:
        output_file = input_file.replace(".jsonl", f"_cat_{revision}.jsonl")
    else:
        output_file = input_file.replace(".jsonl", "_cat.jsonl")
    assert output_file != input_file, f"Invalid input file: {input_file}"

    judge = OpenAIJudge(judge_model)

    skip_length = estimate_skip_length(output_file)
    if skip_length > 0:
        print(f"Skip {skip_length} items")

    def process_item(item):
        question_id = item["sql_idx"] if "sql_idx" in item else item["index"]
        question = questions[question_id]["question"]
        db_id = db_ids[question_id]
        # pred_sql = item.get("pred") or item["predicted_sql"]
        # gt_sql = item.get("gold") or item["ground_truth"]

        ex = item["ex"]

        if "error" in item: pass # error
        elif ex == 1 and item["judge_result"]: pass # true positive
        else: # false positive, false negative or true negative
            result = judge.categorize_error(
                schema_dict[db_id],
                question,
                item["pred_sql"],
                item["pred_exec"],
                item["gt_sql"],
                item["gt_exec"],
                ex=ex,
                llm_judgment=item["judge_text"],
                llm_judge_result=item["judge_result"],
                hint=item["hint"],
                verbose=True,
            )
            item["judge_result_category"] = result[1]
        
        return item
    
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
    print(f"Finished! Output: {output_file}")

    print_result(output_file)


COL_RENAME = {
    'error': "Error",
    'incorrect_schema_linking': "Schema Link.",
    'incorrect_filtering_condition': "Filtering Cond.",
    'missing_handling_of_nullable_column': "Nullable Col.",
    'missing_handling_of_multiple_rows': "Multiple Rows",
    'abused_clauses': "Abused Clauses",
    'other_fatal_issues': "Other",
    "different_output_structure": "Structure",
    "different_output_value_representation": "Value Repr.",
    "incorrect_ground_truth_query": "Ground Truth",
    "multiple_answers_available": "Ambiguity",
    "other_minor_issues": "Other"
}
FP_errors = [
    'error',
    'incorrect_schema_linking',
    'incorrect_filtering_condition',
    'missing_handling_of_nullable_column',
    'missing_handling_of_multiple_rows',
    'abused_clauses',
    'other_fatal_issues'
    ]
keys = {
    "FN": ["different_output_structure","different_output_value_representation","incorrect_ground_truth_query","multiple_answers_available","other_minor_issues"],
    "FP": FP_errors,
    "TN": FP_errors,
}

def error_cat_parse(cat):
    return {k: v["issued"] for k, v in cat.items()}


def print_result(output_file):
    items = list(jsonlines.open(output_file))
    model = output_file.replace("_ex_judge_cat.jsonl", "").split("/")[-1]
    flex = sum([item["judge_result"] == True for item in items]) / len(items)
    rows = []

    for item in items:
        if item["judge_result"] == "error":
            rows.append({
                "model": model,
                "error": True,
                "Type": "TN",
                "FLEX": flex,
            })
        elif item["ex"] == 1 and item["judge_result"]: # True positive
            continue
        elif item["ex"] == 0 and item["judge_result"]: # False negative
            rows.append({
                "model": model,
                "Type": "FN",
                "FLEX": flex,
                **error_cat_parse(item["judge_result_category"])
            })
        else: # False positive or True negative
            rows.append({
                "model": model,
                "Type": "FP" if item["ex"] == 1 else "TN",
                "FLEX": flex,
                **error_cat_parse(item["judge_result_category"])
            })

    df = pd.DataFrame(rows)
    if "error" not in df:
        df["error"] = False
    
    print(f"Model: {model}")

    for error_type in ["FN", "FP", "TN"]:
        subdf = df[df["Type"] == error_type][keys[error_type]].fillna(0)

        sums = subdf.sum()
        if error_type == "FP":
            # drop index "error"
            sums = sums.drop("error")
        
        # rename index
        sums = sums.rename(COL_RENAME)

        print(f"\n-- {error_type} (Total: {len(subdf)}) --")
        print(sums.sort_values(ascending=False))

def main_batch(
    input_file: str,
    dataset: str,
    judge_model: str = "gpt-4o-mini",
    db_engine: str = "sqlite",
    batch_size: int = 1,
    num_threads: int = 1,
    revision: str = None,
    limit: int = None,
):
    files = glob.glob(input_file)
    for file in files:
        main(file, dataset, judge_model, db_engine, batch_size, num_threads, revision, limit)
    
if __name__ == "__main__":
    fire.Fire(main_batch)