import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import os
from pprint import pprint


def load_json(dir):
    with open(dir, 'r') as j:
        contents = json.loads(j.read())
    return contents

def result_callback(result):
    exec_result.append(result)


def execute_sql(predicted_sql,ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    ground_truth_res = None

    try:
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
    except Exception as e:
        print('An error occured while executing a ground truth query:', str(e))
        # print('Query:', ground_truth)
        # print('Database:', db_path)
        raise

    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()

    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    # elif len(set(predicted_res)) == len(set(ground_truth_res)):
    #     print("\n이건 왜 틀렸지")
    #     pprint(predicted_res)
    #     print()
    #     pprint(ground_truth_res)
    #     print("---")
    return res, list(predicted_res), list(ground_truth_res)



def execute_model(predicted_sql,ground_truth, db_place, idx, meta_time_out):
    predicted_res, ground_truth_res = None, None
    error = None
    try:
        res, predicted_res, ground_truth_res = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        print(f'Timeout at {idx}')
        res = 0
        error = 'timeout'
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        print(f'({db_place}) Error at {idx}: {e}')
        res = 0
        error = str(e)
    # print(result)
    # result = str(set([ret[0] for ret in result]))
    result = {'sql_idx': idx, 'res': res, 'predicted_sql': predicted_sql, 'ground_truth': ground_truth, 'db_path': db_place,
              'predicted_res': predicted_res, 'ground_truth_res': ground_truth_res}
    if error:
        result['error'] = error
    # print(result)
    return result


def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev', gt_db_paths=None):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        if sql_path.endswith('.sql'):
            sql_data = [x.strip() for x in open(sql_path, 'r').readlines() if x.strip()]
            sql_data = {idx: sql for idx, sql in enumerate(sql_data)}
        elif os.path.exists(sql_path):
            sql_data = json.load(open(sql_path))
            if "CHESS" in sql_path:
                print("CHESS", len(sql_data))
                sql_data = {int(x["question_id"]): x["predicted_sql"] for x in sql_data}
            print("load from json")
        else:
            sql_data = json.load(open(sql_path + 'predict_' + data_mode + '.json', 'r'))

        for idx in sorted(sql_data.keys()):
            sql_str = sql_data[idx]
            if sql_path.endswith('.sql'):
                sql = sql_str
                db_name = gt_db_paths[idx].split('/')[-2].strip()
            elif type(sql_str) == str:
                if '\t----- bird -----\t' in sql_str:
                    sql, db_name = sql_str.split('\t----- bird -----\t')
                else:
                    sql, db_name = sql_str, ""
            else:
                sql, db_name = " ", "financial"

            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == 'gt':
        sqls = open(sql_path + data_mode + '_gold.sql')
        sql_txt = sqls.readlines()
        # sql_txt = [sql.split('\t')[0] for sql in sql_txt]
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i,sql_pair in enumerate(sqls):

        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results,diff_json_path):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    for i,content in enumerate(contents):
        exec_results[i]['difficulty'] = content['difficulty']
        exec_results[i]['question'] = content['question']
        exec_results[i]['evidence'] = content['evidence']

        if content['difficulty'] == 'simple':
            simple_results.append(exec_results[i])

        if content['difficulty'] == 'moderate':
            moderate_results.append(exec_results[i])

        if content['difficulty'] == 'challenging':
            challenging_results.append(exec_results[i])

    simple_acc = sum([res['res'] for res in simple_results])/len(simple_results)
    moderate_acc = sum([res['res'] for res in moderate_results])/len(moderate_results)
    challenging_acc = sum([res['res'] for res in challenging_results])/len(challenging_results)
    all_acc = sum(results)/num_queries
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists



def print_data(score_lists,count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--mode_gt', type=str, default='gt')
    args_parser.add_argument('--mode_predict', type=str, default='gpt')
    args_parser.add_argument('--difficulty',type=str,default='simple')
    args_parser.add_argument('--diff_json_path',type=str,default='')
    args_parser.add_argument('--save_result',type=bool,default=True)
    args = args_parser.parse_args()
    exec_result = []

    # generate gt sqls:
    gt_queries, db_paths_gt = package_sqls(args.ground_truth_path, args.db_root_path, mode='gt',
                                           data_mode=args.data_mode)
    
    pred_queries, db_paths = package_sqls(args.predicted_sql_path, args.db_root_path, mode=args.mode_predict,
                                          data_mode=args.data_mode, gt_db_paths=db_paths_gt)

    print(len(pred_queries), len(gt_queries))

    query_pairs = list(zip(pred_queries,gt_queries))
    run_sqls_parallel(query_pairs, db_places=db_paths_gt, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)
    
    print(len(exec_result))
    print('start calculate')
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
        compute_acc_by_diff(exec_result,args.diff_json_path)
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists,count_lists)
    print('===========================================================================================')
    print("Finished evaluation")

    if args.save_result:
        incorrects_filename = os.path.splitext(args.predicted_sql_path)[0] + '_ex.json'
        with open(incorrects_filename,'w') as f:
            json.dump(exec_result,f,indent=4)
        print(f'The evaluation result has been saved in {incorrects_filename}')
    