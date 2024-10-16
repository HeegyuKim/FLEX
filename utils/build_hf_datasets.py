from glob import glob
import datasets as hfds
import sqlite3
import json
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
import os

CATEGORY_TOP_K = 30

# def get_table_schema(db):

#     # 데이터베이스 연결
#     conn = sqlite3.connect(db)
#     conn.text_factory = lambda b: b.decode(errors = 'ignore')

#     cursor = conn.cursor()

#     # 모든 테이블 이름 가져오기
#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#     tables = cursor.fetchall()

#     outputs = []

#     # 각 테이블의 스키마 출력
#     for table in tables:
#         table_name = table[0]

#         # print(f"Schema for table {table_name}:")
#         cursor.execute(f'PRAGMA table_info("{table_name}");')

#         schema = cursor.fetchall()
#         column_definitions = []
#         for column in schema:
#             col_def = f"  {column[1]} {column[2]}"
#             if column[3] == 1:
#                 col_def += " NOT NULL"
#             if column[4] is not None:
#                 col_def += f" DEFAULT {column[5]}"
#             if column[5] == 1:
#                 col_def += " PRIMARY KEY"
#             column_definitions.append(col_def)
#         schema_text = f"CREATE TABLE \"{table_name}\" (\n"
#         schema_text += ",\n".join(column_definitions)
#         schema_text += "\n);"

#         df = pd.read_sql_query(f"SELECT * FROM \"{table_name}\"", conn)

#         columns = []

#         for column in df.columns:
#             col_info = {"column_name": column}
#             if pd.api.types.is_numeric_dtype(df[column]):
#                 col_info["type"] = "numeric"
#                 col_info["statistics"] = df[column].describe().to_dict()
#             else:
#                 col_info["top_value_counts"] = df[column].value_counts().head(CATEGORY_TOP_K).to_dict()
#                 col_info["unique_values"] = len(df[column].unique())

#                 if isinstance(df[column].dtype, pd.CategoricalDtype):
#                     col_info["type"] = "categorical"
#                 elif pd.api.types.is_string_dtype(df[column]):
#                     col_info["type"] = "string"
#                 else:
#                     col_info["type"] = "other"

#             columns.append(col_info)

#         cursor.execute(f"SELECT COUNT(*) FROM \"{table_name}\"")
#         record_count = cursor.fetchone()[0]

#         outputs.append({
#             'table_name': table_name,
#             'schema': schema_text,
#             'columns': json.dumps(columns, ensure_ascii=False),
#             'num_rows': record_count
#         })

#     # 연결 종료
#     conn.close()
#     return outputs


def get_constraints_info(database_path):
    # 데이터베이스 연결
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # 모든 테이블의 이름 가져오기
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # 각 테이블의 제약 조건 정보 가져오기
    constraints_info = {}
    for table in tables:
        table_name = table[0]
        
        # 테이블 컬럼 정보 및 NOT NULL, DEFAULT 값 확인
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns_info = cursor.fetchall()
        
        # Primary Key 정보 추출
        primary_keys = [col[1] for col in columns_info if col[5] > 0]  # col[5] is the 'pk' field
        
        # Foreign Key 정보 가져오기
        cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
        foreign_keys_info = cursor.fetchall()
        
        # UNIQUE 제약 조건 확인
        cursor.execute(f"PRAGMA index_list(`{table_name}`);")
        indexes = cursor.fetchall()
        unique_constraints = []
        for index in indexes:
            index_name = index[1]
            if index[2]:  # index[2] is a boolean, true if the index is unique
                cursor.execute(f"PRAGMA index_info(`{index_name}`);")
                unique_info = cursor.fetchall()
                unique_constraints.append(unique_info)
        
        # CHECK 제약 조건 확인 (SQL 쿼리를 통해 분석)
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        create_table_sql = cursor.fetchone()[0]
        check_constraints = []
        if "CHECK" in create_table_sql.upper():
            # 단순히 SQL 문에서 CHECK 제약 조건을 추출
            check_constraints = [line.strip() for line in create_table_sql.splitlines() if "CHECK" in line.upper()]
        
        constraints_info[table_name] = {
            'columns': columns_info,
            'primary_keys': primary_keys,
            'foreign_keys': foreign_keys_info,
            'unique_constraints': unique_constraints,
            'check_constraints': check_constraints,
        }
    
    # 연결 종료
    conn.close()
    
    return constraints_info

def generate_create_table_sql(table_name, constraints_info):
    columns_info = constraints_info['columns']
    primary_keys = constraints_info['primary_keys']
    foreign_keys_info = constraints_info['foreign_keys']
    unique_constraints = constraints_info['unique_constraints']
    check_constraints = constraints_info['check_constraints']

    # 컬럼 정의
    column_definitions = []
    for col in columns_info:
        col_name = col[1]
        col_type = col[2]
        not_null = "NOT NULL" if col[3] else ""
        default_value = f"DEFAULT {col[4]}" if col[4] is not None else ""
        pk = "PRIMARY KEY" if col[5] else ""
        column_def = f"{col_name} {col_type} {not_null} {default_value} {pk}".strip()
        column_definitions.append(column_def)
    
    # Primary Key 제약 조건
    if primary_keys and len(primary_keys) > 1:  # 복합 Primary Key의 경우
        pk_constraint = f"PRIMARY KEY ({', '.join(primary_keys)})"
        column_definitions.append(pk_constraint)
    
    # Foreign Key 제약 조건
    foreign_key_constraints = []
    for fk in foreign_keys_info:
        fk_column = fk[3]
        ref_table = fk[2]
        ref_column = fk[4]
        fk_constraint = f"FOREIGN KEY ({fk_column}) REFERENCES {ref_table}({ref_column})"
        foreign_key_constraints.append(fk_constraint)
    
    # Unique 제약 조건
    unique_constraints_sql = []
    for unique in unique_constraints:
        unique_columns = ", ".join([u[2] for u in unique])  # unique[2] is the column name
        unique_constraint = f"UNIQUE ({unique_columns})"
        unique_constraints_sql.append(unique_constraint)
    
    # Check 제약 조건
    check_constraints_sql = check_constraints
    
    # 전체 제약 조건 결합
    all_constraints = (
        column_definitions +
        unique_constraints_sql +
        foreign_key_constraints +
        check_constraints_sql
    )

    # CREATE TABLE SQL 생성
    create_table_sql = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(all_constraints) + "\n);"
    
    return create_table_sql

def db_file_to_schema(db_file, read_table: bool = False):
    db_name = db_file.split('/')[-1].split('.')[0]
    table_info = get_constraints_info(db_file)
    schema_sqls = []
    csv_files = []

    for table_name, constraints_info in table_info.items():
        schema_sqls.append(generate_create_table_sql(table_name, constraints_info))

        csv_file = os.path.join(os.path.dirname(db_file), "database_description/" + table_name + ".csv")
        if read_table and os.path.exists(csv_file):
            with open(csv_file, encoding="cp1252") as f:
                csv_files.append(f"{table_name} Table Description\n" + f.read())

    schema_sqls = "\n\n".join(schema_sqls)

    

    print(db_file)
    print(schema_sqls)
    print("---")

    return {
        'db_id': db_name,
        'tables': json.dumps(table_info),
        'schema': schema_sqls,
        'description': "\n\n".join(csv_files)
    }

# items = []
# for db_file in tqdm(list(glob('spider/database/*/*.sqlite'))):
#     items.append(db_file_to_schema(db_file))
# ds = hfds.Dataset.from_list(items)
# ds.push_to_hub('iknow-lab/spider-schema')


train_items = []
for db_file in tqdm(list(glob('bird-download/train_databases/*/*.sqlite'))):
    train_items.append(db_file_to_schema(db_file))
    
dev_items = []
for db_file in tqdm(list(glob('bird_dev/llm/data/dev_databases/*/*.sqlite'))):
    dev_items.append(db_file_to_schema(db_file, True))
    
dev_mini_items = []
for db_file in tqdm(list(glob('bird_mini_dev/llm/mini_dev_data/minidev/MINIDEV/dev_databases/*/*.sqlite'))):
    dev_mini_items.append(db_file_to_schema(db_file, True))

dd = hfds.DatasetDict({
    'train': hfds.Dataset.from_list(train_items),
    'dev': hfds.Dataset.from_list(dev_items),
    'dev_mini': hfds.Dataset.from_list(dev_mini_items)
})
dd.push_to_hub('iknow-lab/bird-schema')