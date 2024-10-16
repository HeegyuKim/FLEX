import fire
import os
import json, jsonlines
from utils.db_utils import load_database
from tqdm.auto import tqdm
from utils.llm_judge import get_model
from utils.eval_utils import estimate_skip_length
from utils.dataset_schema import get_schema_text_dict
from utils import prompts
from datasets import load_dataset
import pandas as pd
import concurrent.futures
import threading
from pprint import pprint
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score


def main(
    input_file: str,
    judge_model: str = "gpt-4o-mini",
    db_engine: str = "sqlite",
    batch_size: int = 1,
    num_threads: int = 1,
    revision: str = None,
    limit: int = None,
):
    if input_file.endswith(".json"):
        items = json.load(open(input_file))
    else:
        items = list(jsonlines.open(input_file))
    
    if judge_model.startswith("vllm") or judge_model.startswith("hf"):
        num_threads, batch_size = 1, 1

    judge, judge_model = get_model(judge_model)

    if revision:
        output_file = input_file.replace(".jsonl", f"_{judge_model}_{revision}.jsonl")
    else:
        output_file = input_file.replace(".jsonl", f"_{judge_model}.jsonl")

    skip_length = estimate_skip_length(output_file)
    if skip_length > 0:
        print(f"Skip {skip_length} items")

    def process_item(item):
        if "error" not in item:
            strict = item["ex"] == 1
            
            judge_text, judge_result = judge.judge(
                item["schema"],
                item["question"],
                item["pred_sql"],
                item["pred_exec"],
                item["gt_sql"],
                item["gt_exec"],
                hint=item.get("hint"),
                verbose=True,
                strict=strict,
            )
            item["judge_text"] = judge_text
            item["judge_result"] = judge_result
            item["judge_model"] = judge_model

        return item
    
    fout = jsonlines.open(output_file, "a")
    lock = threading.Lock()

    def process_batch(batch):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_item, item) for item in batch]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        results = [result for result in results if result is not None]
        results = sorted(results, key=lambda x: x['eval_index'])
        with lock:
            for item in results:
                fout.write(item)

    batch = []
    for i, item in enumerate(tqdm(items)):
        if i < skip_length:
            continue
        if limit and i >= limit + skip_length:
            break
        item["eval_index"] = i
        if num_threads == 1:
            item = process_item(item)
            fout.write(item)
        else:
            batch.append(item)
            if len(batch) == batch_size:
                process_batch(batch)
                batch = []
    if batch:
        process_batch(batch)

    fout.close()


if __name__ == "__main__":
    fire.Fire(main)