import fire
import pandas as pd
from pprint import pprint
from sklearn.metrics import cohen_kappa_score
import numpy as np



def main(
    fp: str,
    fn: str,
    is_prometheus: bool = False,
    prometheus_threshold: float = 4,
):
    print(f"False Positive File: {fp}")
    print(f"False Negative File: {fn}")
    
    gt_result = "results/agreement/experts_agreement_200.csv"
    gt = pd.read_csv(gt_result)
    gt = pd.DataFrame({
        "Index": gt["Index"],
        "FP_consensus": gt["FP_consensus"],
        "FN_consensus": gt["FN_consensus"],
    }).astype(int)
    print(gt.sum())
    
    df_fp = pd.read_json(fp, lines=True)
    df_fn = pd.read_json(fn, lines=True)

    if len(df_fp) != 100 or len(df_fn) != 100:
        print(f"Length mismatch: {len(df_fp)} vs {len(df_fn)}")
        return
    
    if is_prometheus:
        df_fp.judge_result = df_fp.judge_result.map(lambda x: x["score"] >= prometheus_threshold if x and x.get("score") else "error")
        df_fn.judge_result = df_fn.judge_result.map(lambda x: x["score"] >= prometheus_threshold if x and x.get("score") else "error")

    df_fp.judge_result = df_fp.judge_result.map({True: 1, False: 0, "error": 2}).fillna(2)
    df_fn.judge_result = df_fn.judge_result.map({True: 1, False: 0, "error": 2}).fillna(2)

    gt_list = np.array(gt.FP_consensus.tolist() + gt.FN_consensus.tolist(), dtype=int)
    pred_list = np.array(df_fp.judge_result.astype(int).tolist() + df_fn.judge_result.astype(int).tolist(), dtype=int)

    kappa = cohen_kappa_score(gt_list, pred_list)
    false_positives = (gt_list == 0) & (pred_list == 1)
    false_negatives = (gt_list == 1) & (pred_list == 0)
    fp = np.mean(false_positives) * 100
    fn = np.mean(false_negatives) * 100
    fp_acc = (gt.FP_consensus == df_fp.judge_result).mean() * 100
    fn_acc = (gt.FN_consensus == df_fn.judge_result).mean() * 100

    final_result = {
        "Kap": f"{kappa * 100:.2f}",
        "Acc": (gt_list == pred_list).mean() * 100,
        "FP": fp,
        "FN": fn,
        "EQ": fp_acc,
        "NEQ": fn_acc,
        "# error": (pred_list == 2).sum(),
    }
    pprint(final_result)

    
if __name__ == "__main__":
    fire.Fire(main)