from . import prompts
from datasets import load_dataset
import sqlite3
import os
import pandas as pd
from traceback import print_exc, format_exc



class BaseSQLEvolver:

    def __init__(self, db_path, client, model, max_refine=3):
        self.db_path = db_path
        self.client = client
        self.model = model
        self.max_refine = max_refine

    def evolve_item(self, item, n):
        """
            Process the item
            return: item
        """
        pass

    def load_dataset_schema(self):
        """
            Load the dataset and schema
            return: dataset
        """
        pass

    def question2sql(self, db_name, question):
        pass

    def execute_sql(self, db_name, sql):
        conn = sqlite3.connect(os.path.join(self.db_path, db_name, f"{db_name}.sqlite"))
        conn.text_factory = lambda b: b.decode(errors = 'ignore')

        df = pd.read_sql_query(sql, conn)

        return df

class SpiderSQLEvolver(BaseSQLEvolver):

    def validate_generate_query_and_refine(self, db_id, schema_text, question, query):
        # Run the evolved query
        try:
            execution_result_df = self.execute_sql(db_id, query)
            execution_message, ok = prompts.execution_result2text(execution_result_df)
        except Exception as e:
            print_exc()
            execution_result_df = None
            execution_message = str(e)
            ok = True

        print(execution_message)

        # 실행결과로 0개를 가져온 쿼리는 스킵하고, 에러가 나거나 결과가 있으면 피드백을 받는다.
        if ok:
            # Ask the LLM to evaluate the query with the execution result
            eval_prompt = prompts.EVAL_USER_PROMPT.format(schema=schema_text, question=question, query=query, execution_result=execution_message)
            outputs = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompts.EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": eval_prompt}
                    ],
                max_tokens=4096,
            )
            evaluation_result = outputs.choices[0].message.content
            eval_dict = prompts.parse_evol_sql_answer(evaluation_result) or None
            
            print(evaluation_result)
        else:
            eval_dict = {}
            evaluation_result = None

        return dict(
            query=query,
            
            execution_result=execution_result_df.to_dict() if execution_result_df is not None else None,
            execution_message=execution_message,
            
            evaluation_result=evaluation_result,
            feedback=eval_dict.get('feedback'),
            result=eval_dict.get('result'),
            refined_query=eval_dict.get('refined_query', None)
        )


    def evolve_item(self, item, n):
        schema_text = self.schema[item['db_id']]
        
        # Steps
        # 1. evol query
        # 2. validate query
        # 3. ask llm whether the result is correct or not, if not, ask for the correct query

        user_prompt = prompts.EVOL_USER_PROMPT.format(schema=schema_text, question=item['question'], query=item['query'])
        outputs = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompts.EVOL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
                ],
            max_tokens=4096,
            n=n
        )
        # print(f"Question: {item['question']}\nQuery: {item['query']}")
        print(user_prompt)
        print("-->")
        evol_outputs = []
        num_correct = 0

        for choice in outputs.choices:
            print(choice.message.content)
            # Parse the yaml output (evolved question & query)
            try:
                yaml_data = prompts.parse_evol_sql_answer(choice.message.content)
                evol_question = yaml_data['question']
                evol_query = yaml_data['query']
                refine_steps = []
                correct_query = None

                for i in range(self.max_refine):
                    print(f"Refine {i+1}")
                    eval_result = self.validate_generate_query_and_refine(item['db_id'], schema_text, evol_question, evol_query)
                    refine_steps.append(eval_result)

                    if eval_result['result'] == 'correct':
                        correct_query = evol_query
                        num_correct += 1
                        break
                    evol_query = eval_result['refined_query']
                    if evol_query is None or evol_query == "":
                        break

                evol_outputs.append(dict(
                    evol_question=evol_question,
                    evol_query=evol_query,
                    refine_steps=refine_steps,
                    correct_query=correct_query
                ))
            except:
                print_exc()

        print(f"Correct: {num_correct}/{n}")
        print("*************")


        return dict(
            db_id=item['db_id'],
            question=item['question'],
            query=item['query'],
            evol_outputs=evol_outputs
        )


    def load_dataset_schema(self):
        
        dataset = load_dataset("xlangai/spider", split="train")
        schema = load_dataset("iknow-lab/spider-schema", split="train")
        schema_dict = {}
        for item in schema:
            schema_dict[item['db_id']] = prompts.schema2text(item)

        self.schema = schema_dict

        return dataset