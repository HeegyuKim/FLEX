import json
import yaml
import re
import pandas as pd
import numbers




"""
## Schema
CREATE TABLE abc (
    id SERIAL PRIMARY KEY,
    ...
)
**Statistics for abc**
- Number of rows: 1000
Numeric columns:
- Column1: min=0, max=100, mean=50, std=10, 25%=25, 50%=50, 75%=75
- ...
Other columns:
- Column2: 10 unique values, top 3 (freqency): 'a' (25) , 'b' (10), 'c' (5)
- ...

Create TABLE ... (
    ...
)

...

"""

TABLE_PROMPT = """{schema}
- Number of rows: {num_rows}"""

STATS_PROMPT = """
**Column Statistics for {table_name}**
Caution: The statistics are based on the full rows, they can be different on the SELECT condition.
"""
def table2text(table_dict, add_statistics=True):
    table_name = table_dict['table_name']
    table_text = TABLE_PROMPT.format(schema=table_dict['schema'], num_rows=table_dict['num_rows'])
    columns = json.loads(table_dict['columns'])

    if columns and add_statistics:
        table_text += STATS_PROMPT.format(table_name=table_name)
        for column in columns:
            if "statistics" in column:
                stats = column['statistics']
                table_text += f"- {column['column_name']}: min={stats['min']}, max={stats['max']}, mean={stats['mean']}, std={stats['std']}, 25%={stats['25%']}, 50%={stats['50%']}, 75%={stats['75%']}\n"
            else:
                topk = column['top_value_counts']
                topk_str = ", ".join([f"{k} ({v})" if len(k) <= 20 else f"{k[:20]} ... {len(k)} chars ({v})" for k, v in topk.items()])
                k = len(topk)

                table_text += f"- {column['column_name']}: {column['unique_values']} unique values, top {k} (frequency): {topk_str}\n"

    return table_text

def schema2text(schema_dict, add_statistics=True):
    tables = [table2text(table, add_statistics=add_statistics) for table in schema_dict['tables']]
    return f"Database: {schema_dict['db_id']}\n" + "\n\n".join(tables)

def get_db_table_names(schema_dataset):
    output = {}
    for item in schema_dataset:
        db_id = item['db_id']
        names = [table['table_name'] for table in item['tables']]
        output[db_id] = names
    return output

EVOL_SYSTEM_PROMPT = """You are an SQL expert. You have to write a evolved version of the following natural language question and SQL query. 
1. Evolve the question to hard level, requiring more specific or more complex SQL skills.
2. Answer the evolved question with a SQL query that satisfies answering the evolved question.
3. The evolved question and sql query must be SELECT query and working with the given schema.
4. SQL query must be valid and executable, satisfying the SQLite3 grammar.
5. Do not use unexisting columns, tables, any other unexisting entities in the schema or unexisting SQLite3 functions.

Your response should be a following yaml format:
```yaml
question: str
query: str
```"""

EVOL_USER_PROMPT = """
**Schema**
{schema}

Question: {question}
Query: {query}
""".strip()

def parse_evol_sql_answer(answer):
    yaml_code = re.search(r"```yaml\n(.*)\n```", answer, re.DOTALL).group(1)
    if yaml_code:
        return yaml.safe_load(yaml_code)
    return None

EVAL_SYSTEM_PROMPT = """You are an SQL expert. You have to evaluate the following SQL query with the given question, schema and execution result.
1. Take a closer look at the execution results, and make sure that the given query meets requirements in the question,
2. Before evaluation, write a feedback about the query evaluation result.
3. Classify the query evaluation result as among 'correct', 'wrong' or 'unsure'.
4. If the query is 'wrong', provide a refined query that satisfies the requirements in the question.

**Query Refinement Rules**
1. Analyze the given database schema carefully. Only use tables and columns that are explicitly defined in the schema.
2. Use only SQL syntax and functions that are supported by SQLite3. Avoid using advanced features or syntax that SQLite3 doesn't support.
3. Do not write comments in SQL query. Only write SQL query.
4. Do not change column names in the output
5. When selecting columns, strictly follow the order defined in the table schema. Do not rearrange the columns in the SELECT statement.
6. When selecting from multiple tables, always use aliases for tables in the format T1, T2, T3, etc., in the order they appear in the query. When selecting from single table, do not use aliases.

**Response Format**
1. Feedback: 
2. Decision (wrap in double square brackets): [[correct]], [[wrong]] or [[unsure]]
3. Refined Query (if the query is wrong):
```sql
...
```
""".strip()

EVAL_SYSTEM_PROMPT_COT = """You are an SQL expert. You have to evaluate the following SQL query with the given question, schema and execution result.
1. Take a closer look at the execution results, and make sure that the given query meets requirements in the question,
2. Before evaluation, write a feedback about the query evaluation result.
3. Classify the query evaluation result as among 'correct', 'wrong' or 'unsure'.
4. If the query is 'wrong', provide a refined query that satisfies the requirements in the question.

**Query Refinement Rules**
1. Analyze the given database schema carefully. Only use tables and columns that are explicitly defined in the schema.
2. Use only SQL syntax and functions that are supported by SQLite3. Avoid using advanced features or syntax that SQLite3 doesn't support.
3. Do not write comments in SQL query. Only write SQL query.
4. Do not change column names in the output
5. When selecting columns, strictly follow the order defined in the table schema. Do not rearrange the columns in the SELECT statement.
6. When selecting from multiple tables, always use aliases for tables in the format T1, T2, T3, etc., in the order they appear in the query. When selecting from single table, do not use aliases.

**Response Format**
1. Feedback: 
2. Decision (wrap in double square brackets): [[correct]], [[wrong]] or [[unsure]]

(Note: if the query is wrong or unsure, you should provide a step-by-step for the refined query)
3. Step-by-Step Thinking:
4. Refined Query:
```sql
...
```
""".strip()

# When the user requests a final query, your response should be a following sql format:
# ```sql
# ...
# ```


EVAL_USER_PROMPT = """
**Schema**
{schema}

**Question**
{question}

**Query**
{query}

**Execution Result**
{execution_result}
""".strip()

EVAL_USER_REFINE_PROMPT = """
I have executed your query and got the following result. Give me a feedback:

**Query**
{query}

**Execution Result**
{execution_result}
""".strip()

EVAL_USER_FINAL_PROMPT = """
Based on our conversation, write a final version of the SQL query that satisfies the requirements in the question.
""".strip()


def format_value(value, max_str_length=50):
    if isinstance(value, numbers.Number):
        return str(value)
    else:
        value = str(value).strip().replace("\n", " ")
        if len(value) > max_str_length:
            return f"{value[:max_str_length]}... {len(value)} chars"
        return value
    
    return str(value)

def dataframe_to_markdown(df, max_str_length=30, max_rows=20):
    headers = "| " + " | ".join(df.columns) + " |"
    separator = "|" + "|".join(["---" for _ in df.columns]) + "|"
    
    rows = []
    if len(df) > max_rows:
        top_rows = df.head(10).apply(lambda row: "| " + " | ".join(format_value(cell, max_str_length) for cell in row) + " |", axis=1)
        bottom_rows = df.tail(10).apply(lambda row: "| " + " | ".join(format_value(cell, max_str_length) for cell in row) + " |", axis=1)
        rows = list(top_rows) + ["| " + " | ".join(["..." for _ in df.columns]) + " |"] + list(bottom_rows)
    else:
        rows = df.apply(lambda row: "| " + " | ".join(format_value(cell) for cell in row) + " |", axis=1)
    
    return "\n".join([headers, separator] + list(rows))


def execution_result2text(execution_result: pd.DataFrame):
    if isinstance(execution_result, str) or execution_result is None or len(execution_result) == 0:
        return "Empty execution result", False
    return dataframe_to_markdown(execution_result) + "\nshape=" + str(execution_result.shape), True

def execution_result2text_set(execution_result: list, max_str_length=30, max_rows=20, deduplicate=True):
    def format_row(row):
        return "| " + " | ".join(format_value(cell, max_str_length) for cell in row) + " |"
    
    if isinstance(execution_result, str) or execution_result is None or len(execution_result) == 0:
        return "Empty execution result", False
    
    if deduplicate:
        execution_result = list(set(tuple(row) for row in execution_result))

    num_rows, num_cols = len(execution_result), len(execution_result[0])

    if num_rows > max_rows:
        execution_result = execution_result[:10] + ["..."] + execution_result[-10:]
        
    text = "\n".join([format_row(row) if row != "..." else row for row in execution_result])

    return text + f"\nshape=({num_rows}, {num_cols})", True


LLM_SQL_SOLVER_CounterExample = '''
Here are two SQL queries, Q1 and Q2.
Your task is to quietly think and determine if the following two SQL queries (Q1 and Q2) are semantic equivalent based on the DATABASE SCHEMA.
Return EQUIVALENT or NOT EQUIVALENT in Answer. If the two queries are not equivalent, then please provide a counter example.

Note: Two SQL queries are semantic equivalent if and only if they return the same output for all possible table contents.

Q1:{Q1}

Q2:{Q2}

The DATABASE SCHEMA: {schema}

Answer:
'''


LLM_SQL_SOLVER_MiniatureAndMull = """Here are two SQL queries, Q1 and Q2 following DATABASE SCHEMA.
Your task is to think and determine if the following two SQL queries (Q1 and Q2) are semantically equivalent or not semantically equivalent.

Q1:```{Q1}```

Q2:```{Q2}```

Note: Two SQL queries are semantically equivalent if and only if they yield identical results for all possible databases.

1. Try one example database and observe the output of Q1 and Q2.

2. If the outputs are identical, can you modify the example database such that the outputs of Q1 and Q2 are not identical?

3. If a counter example exists, return THE ANSWER IS NOT EQUIVALENT after step 2.
   Otherwise, return THE ANSWER IS EQUIVALENT after step 2.

DATABASE SCHEMA: {schema}

Answer:"""


LLM_SQL_SOLVER_ExplainAndCompare = """Here are two SQL queries, Q1 and Q2 following DATABASE SCHEMA.
Your task is to think and determine if the following two SQL queries (Q1 and Q2) are equivalent or not equivalent.

Q1:```{Q1}```

Q2:```{Q2}```

First, explain SQL query Q1 and then explain SQL query Q2. 

Are there SIGNIFICANT logic differences between Q1 and Q2?

If the differences are logically SIGNIFICANT, then the Q1 and Q2 are not equivalent.
Otherwise, Q1 and Q2 are equivalent.

Output either THE ANSWER IS EQUIVALENT or THE ANSWER IS NOT EQUIVALENT at the end. (NEVER BOTH)

The DATABASE SCHEMA: {schema}

Answer:"""
