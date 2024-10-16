from .text2sql import OpenAIText2SQL, TEXT2SQL_USER_PROMPT, TEXT2SQL_USER_HINT_PROMPT
import types
import re
import sqlite3
import json
import os
import pandas as pd
from functools import lru_cache
from . import prompts
import sys
import traceback
from io import StringIO



UNITTEST_TEXT2SQL_SYSTEM_PROMPT = """
1. Write a python function that retrieve the result for the given natural language question and pandas DataFrames.
2. You can import only numpy as np, pandas as pd, and standard python 3.10 libraries.
3. Details of dataframes are provided as a SQLite database schema.
4. Your response should be a python function that takes the dataframes as input and returns the result as a pandas DataFrame.
5. You can write a natural language step-by-step reasoning before writing the function.

Please be careful following list:
1. Do not use unexisting columns, tables, any other unexisting entities in the schema or unexisting SQLite3 functions.
2. Do not explain the SQL query and the python code.
3. Don't over-interpret the question, but return only the columns you must answer.
For example, if the question is "What is the name of the employee with ID 123?", the SQL query should be "SELECT name FROM employees WHERE id = 123", not "SELECT id, name FROM employees WHERE id = 123".
4. 
If the question is "What is the total number of employees in the company ID 123?", the SQL query should be "SELECT COUNT(*) FROM employees WHERE company_id = 123", not "SELECT company_id, COUNT(*) FROM employees".

""".strip()

FUNCTION_FORMAT = """
```python
import numpy as np
import pandas as pd

def get_result({args}) -> pd.DataFrame:
    # your code here
    return result
```"""

CODE_REFINE_PROMPT = """
Here's a result of your code execution. Please carefully check the result with the given question.
{result}

If wrong, please refine your code to get the correct result. 
```python
... # same function structure
```

If correct, write a SQLite query that generates the same result.
```sql
... # same function structure
```

Note: Do not write multiple code blocks in your response, only write one code block for the refined code or the sql query.
"""

SQL_REFINE_PROMPT = """
Here's a result of your SQL execution.
{result}

If wrong, please refine your SQL query to get the correct result. 
```sql
... # same function structure
```
If correct, just say "[[FINISH]]" to finish the task.

Note: Do not write multiple code blocks in your response. Check
"""


POSTPROCESSING_PROMPT = """
Your task is analyze the given SQL query and natural language question and rewrite the SQL query when necessary.
1. If the SQL query is correct and complete, just say "[[correct]]" and do not write the query.
2. The SQL query must selects only the columns that are explicitly mentioned in the question.

Example 1: What is the name of the employee with ID 123?
Correct query: SELECT name FROM employees WHERE id = 123
Incorrect query: SELECT id, name FROM employees WHERE id = 123

Example 2: What is the total number of employees in the company ID 123?
Correct query: SELECT COUNT(*) FROM employees WHERE company_id = 123
Incorrect query: SELECT company_id, COUNT(*) FROM employees

Example 3: What is the difference in salary between the highest and lowest paid employees?
Correct query: SELECT MAX(salary) - MIN(salary) FROM employees
Incorrect query: SELECT MAX(salary), MIN(salary), MAX(salary) - MIN(salary) FROM employees

Example 4: Calcalate A-B, B-C, A-C.
Correct query: SELECT A-B, B-C, A-C FROM employees
Incorrect query: SELECT A-B FROM employees UNION ALL SELECT B-C FROM employees UNION ALL SELECT A-C FROM employees

---
Here's the SQL query and the question:
Question: {question}
```sql
{query}
``` 

Output format:
```sql
# Rewritten SQL query here
```
"""


def extract_shortest_python_code(content):
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        return None
    
    return min(matches, key=len)

def extract_shortest_sql_code(content):
    pattern = r"```sql\n(.*?)\n```"
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        return None
    
    return min(matches, key=len)


class Text2SQLWithCodeFeedback(OpenAIText2SQL):
    def __init__(self, db_path: str, db_table_names: list[str], model: str, base_url: str = None, schema_dict: dict = None,
                 num_refine: int = 1, verbose=False):
        super().__init__(model=model, base_url=base_url, schema_dict=schema_dict)
        self.db_path = db_path
        self.db_table_names = db_table_names
        self.num_refine = num_refine
        self.verbose = verbose
        assert self.num_refine > 0, "Number of refinements must be greater than 0."

    @lru_cache(maxsize=1)
    def get_all_tables(self, db_id, deduplication=True):
        conn = sqlite3.connect(os.path.join(self.db_path, db_id, f"{db_id}.sqlite"))
        conn.text_factory = lambda b: b.decode(errors = 'ignore')
        
        outputs = {}
        for table in self.db_table_names[db_id]:
            if " " in table:
                table = f"`{table}`"

            df = pd.read_sql_query(f"SELECT * from {table}", conn)
            if deduplication:
                df = df.drop_duplicates()
            outputs[table] = df

        return outputs

    def execute_code_with_tables(self, python_code, tables, deduplication=True):
        mod = types.ModuleType("dynamic_module")
        try:
            exec(python_code, mod.__dict__)
            get_result = mod.get_result(**tables)
        except:
            get_result = traceback.format_exc()

        if isinstance(get_result, pd.DataFrame) and deduplication:
            get_result = get_result.drop_duplicates()
        return get_result
    
    def execute_sql(self, db_name, sql):
        conn = sqlite3.connect(os.path.join(self.db_path, db_name, f"{db_name}.sqlite"))
        conn.text_factory = lambda b: b.decode(errors = 'ignore')

        try:
            df = pd.read_sql_query(sql, conn)
        except Exception as e:
            return str(e)

        return prompts.execution_result2text(df)[0]
    
    @lru_cache
    def schema2function(self, db_id: str):
        args = []
        for table in self.db_table_names[db_id]:
            if " " in table:
                table = f"`{table}`"
            args.append(f"{table}: pd.DataFrame")
        function = FUNCTION_FORMAT.format(args=", ".join(args))

        return function

    def question2code(self, db_id, question, hint=None):
        if not self.verbose:
            print = lambda *args, **kwargs: None
        else:
            print = __builtins__['print']

        schema_text = self.schema[db_id]
        if hint:
            user_prompt = TEXT2SQL_USER_PROMPT.format(schema=schema_text, question=question)
        else:
            user_prompt = TEXT2SQL_USER_HINT_PROMPT.format(schema=schema_text, question=question, hint=hint)

        user_prompt += "\n\n**Output Function Format**\n" + self.schema2function(db_id)

        messages = [
            {"role": "system", "content": UNITTEST_TEXT2SQL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
            ]
        
        outputs = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.0,
        )
        content = outputs.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        print(content)

        python_code = extract_shortest_python_code(content)
        if python_code is None:
            return None, None
        
        # Get the output
        tables = self.get_all_tables(db_id)
        exec_result = self.execute_code_with_tables(python_code, tables)

        result_markdown = prompts.execution_result2text(exec_result)[0]
        print("코드 실행결과")
        print(result_markdown)

        
        sql_query = None

        # Phase 2: Refine the code
        for i in range(self.num_refine):
            print(f"Code Refine {i+1}")
            refine_prompt = CODE_REFINE_PROMPT.format(result=result_markdown)
            messages.append({"role": "user", "content": refine_prompt})

            outputs = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
            )
            content = outputs.choices[0].message.content
            print(content)
            
            messages.append({"role": "assistant", "content": content})
            
            refined_python_code = extract_shortest_python_code(content)
            sql_query = extract_shortest_sql_code(content)

            if sql_query:
                break

            if refined_python_code:
                python_code = refined_python_code
                exec_result = self.execute_code_with_tables(python_code, tables)
                result_markdown = prompts.execution_result2text(exec_result)[0]
                print("코드 실행결과")
                print(result_markdown)
            else:
                break

        return messages, python_code, result_markdown, sql_query

    def text2sql(self, db_id, question, hint=None):
        if not self.verbose:
            print = lambda *args, **kwargs: None
        else:
            print = __builtins__['print']
        
        messages, _, _, sql_query = self.question2code(db_id, question, hint)
        if sql_query is None:
            return messages, None
    
        result_markdown = self.execute_sql(db_id, sql_query)
        print("SQL 실행결과")
        print(result_markdown)
        
        # Phase 3: Get the final query
        for i in range(self.num_refine):
            print(f"SQL Refine {i+1}")
            refine_prompt = SQL_REFINE_PROMPT.format(result=result_markdown)

            messages.append({"role": "user", "content": refine_prompt})

            outputs = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
            )
            content = outputs.choices[0].message.content
            print(content)
            
            messages.append({"role": "assistant", "content": content})
            if "[[FINISH]]" in content:
                break

            sql_query = extract_shortest_sql_code(content)
            if sql_query and i + 1 != self.num_refine:
                result_markdown = self.execute_sql(db_id, sql_query)
                print("SQL 실행결과")
                print(result_markdown)
            else:
                return None, None

        if sql_query:
            post_processing_message, sql_query = self.sql_post_processing(question, sql_query)
            print("후처리")
            print(post_processing_message)

        return messages, sql_query

    def sql_post_processing(self, question, sql):
        user_prompt = POSTPROCESSING_PROMPT.format(question=question, query=sql)
        messages = [
            {"role": "user", "content": user_prompt}
            ]
        
        outputs = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.0,
        )
        content = outputs.choices[0].message.content
        if "[[correct]]" in content:
            return content, sql
        else:
            sql = extract_shortest_sql_code(content)
        return content, sql