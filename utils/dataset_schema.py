
from datasets import load_dataset, Dataset
from glob import glob
import pandas as pd

def get_schema_text_dict(dataset: str, use_stats: bool = False, db_path: str = None, add_description: bool = True):
    
    if dataset in ["spider"]:
        # schema = load_dataset("iknow-lab/spider-schema", split="train")
        schema = Dataset.from_pandas(pd.read_parquet("utils/schema/spider.parquet"))
        schema_dict = {}
        for item in schema:
            schema_dict[item['db_id']] = item['schema']
    elif dataset.startswith("bird-dev"):
        # schema = load_dataset("iknow-lab/bird-schema", split="dev")
        schema = Dataset.from_pandas(pd.read_parquet("utils/schema/bird-dev.parquet"))
        schema_dict = {}
        
        for item in schema:
            db_id = item['db_id']
            schema_dict[db_id] = item['schema']
            if add_description:
                schema_dict[db_id] += "\n\n**Table Description**\n" + item['description']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return schema_dict

