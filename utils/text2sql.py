import re
import openai
from retry import retry
import os
import pandas as pd
import sqlite3
from . import prompts
from traceback import print_exc
from copy import deepcopy
import yaml

class BaseText2SQL:

    def text2sql(self, db_id, question, hint=None):
        raise NotImplementedError("text2sql method is not implemented")
    
    def text2sql_batch(self, db_ids, questions, hints=None):
        return [self.text2sql(db_id, question, hint=hint) for db_id, question, hint in zip(db_ids, questions, hints)]
    

TEXT2SQL_SYSTEM_PROMPT = """Your task is to convert natural language questions into valid SQL queries based on the provided database schema. Follow these guidelines strictly:

1. Analyze the given database schema carefully. Only use tables and columns that are explicitly defined in the schema.
2. Use only SQL syntax and functions that are supported by SQLite3. Avoid using advanced features or syntax that SQLite3 doesn't support.
3. Do not write comments in SQL query. Only write SQL query.
4. Do not change column names in the output
5. When selecting columns, strictly follow the order defined in the table schema. Do not rearrange the columns in the SELECT statement.
6. When selecting from multiple tables, always use aliases for tables in the format T1, T2, T3, etc., in the order they appear in the query. When selecting from single table, do not use aliases.

Correct:
SELECT id, name, department FROM employees 
SELECT T1.id, T1.name, T2.department FROM employees T1 JOIN departments T2 ON T1.dept_id = T2.id

Incorrect: 
SELECT T1.id, T1.name, T2.department FROM employees T1
SELECT T1.name, T1.id, T2.department AS dept_name FROM employees AS T1 JOIN departments AS T2 ON T1.dept_id = T2.id

Finally, your response should be a following format:
```sql
...
```"""

TEXT2SQL_USER_PROMPT = """
**Schema**
{schema}

**Question**
{question}"""

TEXT2SQL_USER_HINT_PROMPT = """
**Schema**
{schema}

**Question**
{question}

**Hint**
{hint}"""


def parse_text2sql_answer(answer):
    sql_code = re.search(r"```sql\n(.*)\n```", answer, re.DOTALL)
    if sql_code:
        return sql_code.group(1)
    else:
        return None

class OpenAIText2SQL(BaseText2SQL):

    def __init__(self, model: str, base_url: str = None, schema_dict: dict = None):
        self.model = model
        self.schema = schema_dict
        self.client = openai.OpenAI(
            base_url=base_url
        )

    def text2sql(self, db_id, question, hint=None):
        schema_text = self.schema[db_id]
        if hint:
            user_prompt = TEXT2SQL_USER_PROMPT.format(schema=schema_text, question=question)
        else:
            user_prompt = TEXT2SQL_USER_HINT_PROMPT.format(schema=schema_text, question=question, hint=hint)

        output, sql_query = self.run_chat([
                        {"role": "system", "content": TEXT2SQL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                        ])

        return output, sql_query

    def run_chat(self, messages, parse_func=parse_text2sql_answer):
        messages = deepcopy(messages)

        for i in range(3):
            content = None
            try:
                outputs = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.0,
                )
                content = outputs.choices[0].message.content
                sql_query = parse_func(content)
                break
            except yaml.scanner.ScannerError as e:
                from traceback import print_exc
                print_exc()
                if content:
                    print(content)

                if i < 2:
                    print(f"Retrying... {i + 1}/3")
                else:
                    raise
                
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Parsing error occurred. Please try again.\n{e}"})
                continue
            
        return content, sql_query

class ExecutionFeedbackWrapper(BaseText2SQL):


    def __init__(self, db_path, openai_client, openai_model, t2s_model, max_refine: int = 3, verbose: bool = False):
        self.db_path = db_path
        self.openai_client = openai_client
        self.openai_model = openai_model
        self.t2s_model = t2s_model
        self.max_refine = max_refine
        self.verbose = verbose
        self.schema = t2s_model.schema
        

    def text2sql(self, db_id, question, hint=None):
        output, query = self.t2s_model.text2sql(db_id, question, hint=hint)
        schema_text = self.schema[db_id]
        refined_query, refine_steps = self.feedback_and_refine(db_id, schema_text, question, query, max_refine=self.max_refine)
        return refine_steps, refined_query

    def execute_sql(self, db_name, sql):
        conn = sqlite3.connect(os.path.join(self.db_path, db_name, f"{db_name}.sqlite"))
        conn.text_factory = lambda b: b.decode(errors = 'ignore')

        df = pd.read_sql_query(sql, conn)

        return df
    
    def feedback_and_refine(self, db_id, schema_text, question, query, max_refine: int = 3):
        refine_steps = []
        refined_query = query
        correct_query = query

        if self.verbose:
            print(f"Question: {question}")
            print(f"Initial query: {query}")
        
        history_messages = None

        for turn in range(max_refine):
            if self.verbose:
                print(f"Refine {turn+1}")
            eval_result = self.validate_query_and_refine(db_id, schema_text, question, refined_query, history_messages)
            refine_steps.append(eval_result)
            refined_query = eval_result['refined_query'] 

            history_messages = eval_result.pop('history_messages')
            
            if refined_query:
                correct_query = refined_query
            # 주어진 쿼리가 적합하다고 판단하면 refined_query가 없다.
            else:
                break

            if eval_result['result'] == 'correct':
                break
            
        if turn > 0:
            history_messages.append({"role": "user", "content": prompts.EVAL_USER_FINAL_PROMPT})
            _, final_query = self.t2s_model.run_chat(history_messages, parse_func=parse_text2sql_answer)
        else:
            final_query = correct_query

        return final_query, refine_steps
    
    def validate_query_and_refine(self, db_id, schema_text, question, query, history_messages=None, pass_no_result: bool = False):
        # Run the evolved query
        try:
            execution_result_df = self.execute_sql(db_id, query)
            execution_message, ok = prompts.execution_result2text(execution_result_df)
        except Exception as e:
            print_exc()
            execution_result_df = None
            execution_message = str(e)
            ok = True

        if self.verbose:
            print(execution_message)

        # 실행결과로 0개를 가져온 쿼리는 스킵하고, 에러가 나거나 결과가 있으면 피드백을 받는다.
        if not pass_no_result or ok:
            # Ask the LLM to evaluate the query with the execution result
            if history_messages:
                messages = deepcopy(history_messages)
                messages.append({"role": "user", "content": self.build_refine_prompt(schema_text, question, query, execution_message)})
            else:
                messages=[
                    {"role": "system", "content": self.build_system_prompt(schema_text, question, query, execution_result_df)},
                    {"role": "user", "content": self.build_feedback_prompt(schema_text, question, query, execution_result_df)},
                    ]
            
            evaluation_result, refined_query = self.t2s_model.run_chat(messages)
            messages.append({"role": "assistant", "content": evaluation_result})

            if self.verbose:
                print(evaluation_result)
        else:
            evaluation_result = None

        # if "[[correct]]" in evaluation_result:
        #     result = "correct"
        # elif "[[incorrect]]" in evaluation_result:
        #     result = "incorrect"
        # elif "[[wrong]]" in evaluation_result:
        #     result = "wrong"
        result = re.search(r"\[\[(correct|incorrect|unsure)\]\]", evaluation_result)
        if result:
            result = result.group(1)
        else:
            result = "parse-error"

        return dict(
            query=query,
            
            execution_result=execution_result_df.to_dict() if execution_result_df is not None else None,
            execution_message=execution_message,
            
            evaluation_result=evaluation_result,
            result=result,
            refined_query=refined_query,
            history_messages=messages
        )
    
    def build_system_prompt(self, schema_text, question, query, execution_result):
        return prompts.EVAL_SYSTEM_PROMPT
    
    def build_feedback_prompt(self, schema_text, question, query, execution_result):
        return prompts.EVAL_USER_PROMPT.format(schema=schema_text, question=question, query=query, execution_result=execution_result)

    def build_refine_prompt(self, schema_text, question, query, execution_message):
        return prompts.EVAL_USER_REFINE_PROMPT.format(query=query, execution_result=execution_message)
                
class HuggingfaceLM(BaseText2SQL):
    def __init__(self) -> None:
        super().__init__()


    def text2sql(self, db_id, question, hint=None):
        raise NotImplementedError("text2sql method is not implemented")
    

class VLLMWrapper(BaseText2SQL):
    def __init__(self, model_name: str, device: str = "cuda", dtype = "bfloat16", max_length: int = 4096, chat_template: str = None):
        from vllm import LLM, SamplingParams
        if "@" in model_name:
            from huggingface_hub import snapshot_download
            model_name, revision = model_name.split("@")
            model_name = snapshot_download(repo_id=model_name, revision=revision)

        self.model = LLM(model=model_name, dtype=dtype, device=device, max_model_len=max_length)
        self.tokenizer = self.model.get_tokenizer()
        
        self.stop_tokens = list(set([self.tokenizer.eos_token, "<|endoftext|>", "[INST]", "[/INST]", "<|im_end|>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<start_of_turn>", "</s>"]))

        # if chat_template:
        #     self.tokenizer.chat_template = PROMPT_TEMPLATES[chat_template]

        self.sampling_params = SamplingParams(
            temperature=0,
            max_length=max_length,
            stop=self.stop_tokens,
        )

    def generate_batch(self, prompts, histories=None, generation_prefix: str = None, gen_args={}):
        if histories is None:
            histories = [[] for _ in prompts]
        else:
            histories = deepcopy(histories)

        final_prompts = []
        for prompt, history in zip(prompts, histories):
            history.append({
                'role': 'user',
                'content': prompt,
            })

            inputs = self.tokenizer.apply_chat_template(history, add_special_tokens=True, tokenize=False, add_generation_prompt=True)
            if generation_prefix is not None:
                inputs = generation_prefix + inputs
            
            final_prompts.append(inputs)
        
        sampling_params = self.gen_args_to_sampling_params(gen_args)
        outputs = self.model.generate(final_prompts, sampling_params, use_tqdm=False)
        
        return [output.outputs[0].text for output in outputs]
    
    def generate(self, prompt, history=None, generation_prefix: str = None, gen_args={}):
        if history is None:
            history = []
        else:
            history = deepcopy(history)

        history.append({
            'role': 'user',
            'content': prompt,
        })

        inputs = self.tokenizer.apply_chat_template(history, add_special_tokens=True, tokenize=False, add_generation_prompt=True)
        if generation_prefix is not None:
            inputs = inputs + generation_prefix
        
        sampling_params = self.gen_args_to_sampling_params(gen_args)
        outputs = self.model.generate([inputs], sampling_params, use_tqdm=False)
        
        return outputs[0].outputs[0].text

    def build_prompt(self, schema_text, question, hint=None):
        if hint:
            prompt = TEXT2SQL_USER_HINT_PROMPT.format(schema=schema_text, question=question, hint=hint)
        else:
            prompt = TEXT2SQL_USER_PROMPT.format(schema=schema_text, question=question)
        return prompt

    def text2sql(self, db_id, question, hint=None):
        raise NotImplementedError("text2sql method is not implemented")
    
    def text2sql_batch(self, db_ids, questions, hints=None):
        prompts = [self.build_prompt(self.schema[db_id], question, hint=hint) for db_id, question, hint in zip(db_ids, questions, hints)]
        outputs = self.llm.generate(prompts)
        return [x.outputs[0].text for x in outputs]
    

COT_TEXT2SQL_SYSTEM_PROMPT = """Your task is to convert natural language questions into valid SQL queries based on the provided database schema. Follow these guidelines strictly:

1. Analyze the given database schema carefully. Only use tables and columns that are explicitly defined in the schema.
2. Use only SQL syntax and functions that are supported by SQLite3. Avoid using advanced features or syntax that SQLite3 doesn't support.
3. Do not write comments in SQL query. Only write SQL query.
4. Do not change column names in the output
5. When selecting columns, strictly follow the order defined in the table schema. Do not rearrange the columns in the SELECT statement.
6. When selecting from multiple tables, always use aliases for tables in the format T1, T2, T3, etc., in the order they appear in the query. When selecting from single table, do not use aliases.

Correct:
SELECT id, name, department FROM employees 
SELECT T1.id, T1.name, T2.department FROM employees T1 JOIN departments T2 ON T1.dept_id = T2.id

Incorrect: 
SELECT T1.id, T1.name, T2.department FROM employees T1
SELECT T1.name, T1.id, T2.department AS dept_name FROM employees AS T1 JOIN departments AS T2 ON T1.dept_id = T2.id

Finally, your response should be a following format after step-by-step thinking:
# your thinking here

# SQL query here
```sql
SELECT ..
```"""
class CoTText2SQL(OpenAIText2SQL):

    def __init__(self, model: str, base_url: str = None, schema_dict: dict = None):
        super().__init__(model, base_url, schema_dict)

    def text2sql(self, db_id, question, hint=None):
        schema_text = self.schema[db_id]
        user_prompt = TEXT2SQL_USER_HINT_PROMPT.format(schema=schema_text, question=question, hint=hint)
        output, sql_query = self.run_chat([
                        {"role": "system", "content": COT_TEXT2SQL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                        ])

        return output, sql_query
    

class CoTExecutionFeedbackWrapper(ExecutionFeedbackWrapper):

    def build_system_prompt(self, schema_text, question, query, execution_result):
        return prompts.EVAL_SYSTEM_PROMPT_COT
    

