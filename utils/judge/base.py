
import json
import openai
import os
import anthropic
from pathlib import Path
import google.generativeai as genai
import retry
import requests

SYSTEM_STRICT_SIMPLE = """The Prediction Result matches the Reference Result. However, this does not guarantee that the Prediction Query is correct. Carefully analyze the Prediction Query and evaluate its correctness considering the following criteria:

Correct Prediction Query
- If the Prediction Query missed some tables or columns, it is acceptable if the missing information does not affect the query's ability to answer the Question.
- missing DISTINCT in SELECT is always allowed because we do not consider duplicate rows.

Incorrect Prediction Query
- Does not logically answer the Question or contains significant errors.
- Produces different results due to incorrect filering or missing conditions, JOIN redundancy, or other fatal issues.
- Fails to handle null values, multiple rows, or other critical aspects of the Question.
- Does not consider nullable columns in aggregation functions (SUM, COUNT, AVG) and NULL value can lead to unexpected results.
For example, COUNT(*) in the prediction but COUNT(school) in the ground truth will produce different results if the column school does not have a NOT NULL constraint in the schema.
- Does not produce correct results when multiple rows satisfy the condition (e.g min/max, multiple transactions in a day).
- Abused clauses (LIMIT, GROUP BY) to limit the results when the user didn't request it.

SQLite3 Compatibility
- Both queries are SQLite3 compatible, meaning integer division is not automatically converted to float division, and logical operators, column/table names are case-insensitive.
- comparison between string and integer is allowed in SQLite3.

Analysis Guidelines
1. Compare the Prediction Query with the Ground Truth Query within the context of the provided schema and question.
2. Predict the query's logical correctness based on the criteria mentioned above.

Finally, score the Prediction Query as follows:
```json
{"correct": true of false}
```
""".strip()

SYSTEM_LOOSE_SIMPLE = """The Prediction Result differs from the Ground Truth Result. However, this does not necessarily mean that the Prediction Query is incorrect. Analyze the differences between the Prediction Query and Reference Query, considering the following:

Correct
- The Prediction Query logically answers the Question, even if the output structure differs from the Ground Truth Query.
- Do not consider column naming, column/row ordering.
- Some extra column or missing column in the output structure is acceptable if it does not affect the query's ability to answer the Question.
- Differences in the representation of values, such as formatting (percentile, YES/NO) or data types, are acceptable if they do not affect the query's logical correctness.
- Ambiguous questions may have multiple correct answers, so the Prediction Query may differ from the Ground Truth Query.
- Multiple rows are acceptable when the calculate the min/max

Incorrect
- The Prediction Query does not logically answer the Question or contains significant errors.
- The Prediction Query produces different results due to incorrect filering or missing conditions, JOIN redundancy, or other fatal issues.
- The Prediction Query fails to handle null values, multiple rows, or other critical aspects of the Question.
- The result of the Prediction Query is significantly different from the Ground Truth Query even its structure is similar.

SQLite3 Compatibility
- Both queries are SQLite3 compatible, meaning integer division is not automatically converted to float division, and logical operators are case-insensitive.
- If the table schema and description are different, follow the schema provided in the prompt.

Provide a detailed comparison of the Prediction Query and Ground Truth Query, focusing on the nature and significance of their differences. If the Prediction Query is incorrect, explain the specific errors and how they affect the query's ability to answer the Question.

Finally, score the Prediction Query as follows:
```json
{"correct": true/false}
```
""".strip()


ERROR_CATEGORY_FP_TN = """
You are an expert SQL evaluator responsible for assessing the correctness and quality of SQL queries. 
Analyze the judgment thoroughly and provide a categorized evaluation based on the following criteria:
Two queries have no compile-time errors, which means they are both valid SQL queries, integer division is not automatically converted to float division, and logical operators are case-insensitive.

### Criteria
1. Incorrect Schema Linking: Utilized tables and columns in the predicted query do not align with the question and the provided schema. Different columns are permissible if they are described as equivalent in the table description or if the question does not specify exact column names.
2. Incorrect Filtering Conditions: The prediction query incorrectly filters data based on the given conditions, ensuring the WHERE clause is used appropriately to match expected results.
3. Missing Handling of Nullable Column: Check if the query correctly handles nullable columns in aggregation functions (SUM, COUNT, AVG) or other operations, as improper handling can lead to unexpected results. Do not consider NULL values in arithmetic operations.
4. Missing Handling of Multiple Rows: Determine if the query correctly accounts for scenarios where multiple rows might satisfy the condition (e.g min/max, multiple transactions in a day), potentially leading to incorrect answers. Consider primary and foreign keys, as well as unique constraints, which ensure uniqueness.
5. Abused Clauses: Evaluate if SQL clauses like GROUP BY, HAVING, ORDER BY, and DISTINCT are used unnecessarily, which could produce incorrect results.
6. Other Fatal Logical Issues: Identify any additional logical problems not covered by the above criteria.


### Output Format
NOTE: If no issue is found, the explanation can be an empty string. Output only the JSON object containing your evaluation results. Make sure the JSON is properly formatted and valid, with all boolean values in lowercase (true/false) and all strings properly enclosed in double quotes.

```json
{
    "incorrect_schema_linking": {
        "issued": true/false,
        "explanation": "Your explanation here"
    },
    "incorrect_filtering_condition": {
        "issued": true/false,
        "explanation": "Your explanation here"
    },
    "missing_handling_of_nullable_column": {
        "issued": true/false,
        "explanation": "Your explanation here"
    },
    "missing_handling_of_multiple_rows": {
        "issued": true/false,
        "explanation": "Your explanation here"
    },
    "abused_clauses": {
        "issued": true/false,
        "explanation": "Your explanation here"
    },
    "other_fatal_issues": {
        "issued": true/false,
        "explanation": "Your explanation here"
    }
}
```
""".strip()
ERROR_CATEGORY_FN = """

You are an expert SQL evaluator responsible for assessing the correctness and quality of SQL queries. 
Analyze the judgment thoroughly and provide a categorized evaluation based on the following criteria:
Two queries have no compile-time errors, which means they are both valid SQL queries, integer division is not automatically converted to float division, and logical operators are case-insensitive.

### Criteria
1. Different Output Structure: Column selection, ordering is different, but should not consider output column naming.
2. Different Output Value Representation: Differences in value representation, such as formatting or data types, should not be fatal unless they affect the logical correctness of the query.
3. Incorrect Ground Truth Query: The ground truth query is incorrect and does not logically answer the question.
4. Mutiple Answers Available: The question is ambiguous and has multiple correct answers.
5. Other minor issues: Other minor issues that do not affect the logical correctness of the query.

### Output Format
NOTE: If no issue is found, the explanation can be an empty string. Output only the JSON object containing your evaluation results. Make sure the JSON is properly formatted and valid, with all boolean values in lowercase (true/false) and all strings properly enclosed in double quotes.

```json
{
    "different_output_structure": {
        "issued": true/false,
        "explanation": "Your explanation here"
    },
    "different_output_value_representation": {
        "issued": true/false,
        "explanation": "Your explanation here"
    },
    "incorrect_ground_truth_query": {
        "issued": true/false,
        "explanation": "Your explanation here"
    },
    "multiple_answers_available": {
        "issued": true/false,
        "explanation": "Your explanation here"
    },
    "other_minor_issues": {
        "issued": true/false,
        "explanation": "Your explanation here"
    }
""".strip()

DB_HINTS = """Schema Description:
- schools.FundingType and frpm.`Charter Funding Type` are not equivalent
- frpm.`County Name` and schools.County are equivalent
- schools.DOC and schools.DOCType are equivalent
- assume precomputed data is always correct
"""

JUDGE_PROMPT_LOOSE = """
**Question**
{question}

**Prediction Query**
{pred_query}

**Prediction Result**
{pred_result}

**Ground Truth Query**
{gt_query}

**Ground Truth Result**
{gt_result}
""".strip()

JUDGE_PROMPT_STRICT = """
**Question**
{question}

**Prediction Query**
{pred_query}

**Ground Truth Query**
{gt_query}

**Execution Result**
{gt_result}
""".strip()

JUDGE_CATEGORIZATION = """
**Question**
{question}

**Prediction Query**
{pred_query}

**Ground Truth Query**
{gt_query}

**Judgment**
{judgement}
""".strip()

JUDGE_PROMPT_STRICT_NORESULT = """
**Question**
{question}

**Prediction Query**
{pred_query}

**Ground Truth Query**
{gt_query}
""".strip()

class LLMJudge():
    USE_HINT = True
    INCLUDE_EX_HINT = True
    
    def get_judge_prompt(self, strict: bool):
        return JUDGE_PROMPT_STRICT_NORESULT if strict else JUDGE_PROMPT_LOOSE

    def get_system_prompt(self, strict: bool):
        return SYSTEM_STRICT_SIMPLE if strict else SYSTEM_LOOSE_SIMPLE

    def judge(self, schema: str, question: str, pred_query, pred_result: str, gt_query, gt_result: str, hint: str = None, verbose: bool = False, strict: bool = True) -> str:
        SYSTEM_PROMPT = self.get_system_prompt(strict)
        JUDGE_PROMPT = self.get_judge_prompt(strict)
            
        prompt = JUDGE_PROMPT.format(
            schema=schema,
            question=question,
            pred_query=pred_query,
            gt_query=gt_query,
            pred_result=pred_result,
            gt_result=gt_result
        )

        if self.INCLUDE_EX_HINT:
            prompt += f"\n\nNote: two queries have {'same' if strict else 'different'} execution results.\n"
        if self.USE_HINT:
            prompt += "\n**Hint**\n" + DB_HINTS + (hint or "")

        content = self.request_llm(SYSTEM_PROMPT, prompt)

        if verbose:
            print(prompt.split('**Question**', 1)[1])
            print("-->", content)
            print("---\n")

        if "```json" in content:
            json_result = content.split("```json\n")[1].split("\n```")[0]
            judgement = json.loads(json_result)
        elif "\"correct\": true" in content:
            judgement = {"correct": True}
        elif "\"correct\": false" in content:
            judgement = {"correct": False}
        else:
            # raise ValueError(f"Invalid response: {content}")
            return content, "error"
            
        return content, judgement['correct']
      
    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        pass

    def categorize_error(self, 
        schema: str,
        question: str,
        pred_query: str,
        pred_result: str,
        gt_query: str,
        gt_result: str,
        ex: int,
        llm_judgment: str,
        llm_judge_result: bool,
        hint: str = None,
        verbose: bool = False,
        ):
        if ex == 1 and llm_judge_result:
            return None # true positive
        elif ex == 0 and llm_judge_result: # false negative
            SYSTEM_PROMPT = ERROR_CATEGORY_FN
        else: # false positive or true negative
            SYSTEM_PROMPT = ERROR_CATEGORY_FP_TN

        prompt = JUDGE_CATEGORIZATION.format(
            schema=schema,
            question=question,
            pred_query=pred_query,
            gt_query=gt_query,
            pred_result=pred_result,
            gt_result=gt_result,
            judgement=llm_judgment
        )

        # prompt += f"\n\nNote: two queries have {'same' if ex == 1 else 'different'} execution results.\n"
        if hint:
            prompt += "\n**Hint**\n" + (hint or "")

        content = self.request_llm(SYSTEM_PROMPT, prompt)

        if verbose:
            print(prompt.split('**Question**', 1)[1])
            print("-->", content)
            print("---\n")

        if "```json" in content:
            json_result = content.split("```json\n")[1].split("\n```")[0]
            try:
                judgement = json.loads(json_result)
            except json.JSONDecodeError:
                judgement = {"llm_error": "Invalid JSON format"}
        else:
            # return content, "error"
            raise ValueError(f"Invalid response: {content}")
        
        return content, judgement
        
            
        

    
class OpenAIJudge(LLMJudge):
    def __init__(self, model: str, base_url: str = None, api_key: str = None) -> None:
        self.model = model
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.max_tokens = 2048

    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        if "gemma" in self.model or self.model.startswith("o1"):
            messages = [
                {"role": "user", "content": system_prompt + "\n\n" + user_prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": user_prompt}
                ]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
        if self.model.startswith("o1"):
            kwargs = {
                "max_completion_tokens": self.max_tokens,
            }
        else:
            kwargs = {
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
            }

        outputs = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        content = outputs.choices[0].message.content
        return content

class GeminiJudge(LLMJudge):
    def __init__(self, model: str = 'gemini-1.5-flash') -> None:
        super().__init__()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        self.model = genai.GenerativeModel(model)

    @retry.retry(tries=3, delay=1)
    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.model.generate_content(prompt)
        return response.text

class AnthropicJudge(LLMJudge):
    def __init__(self, model: str = "claude-3-5-sonnet-20240620") -> None:
        super().__init__()
        self.client = anthropic.Anthropic()
        self.model = model

    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
        )
        return message.content[0].text

class MistralAIJudge(LLMJudge):

    def __init__(self, model: str = "mistral-large-2409") -> None:
        super().__init__()
        from mistralai import Mistral
        api_key = os.environ["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=api_key)
        self.model = model
        print(f"Loading Mistral model {model}")

    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        chat_response = self.client.chat.complete(
            model = self.model,
            max_tokens=2048,
            temperature=0,
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        )

        return chat_response.choices[0].message.content
    
    
class HfJudge(LLMJudge):
    
    def __init__(self, model: str, max_seq_length: int = 8192) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.model_name = model
        print(f"Loading model {model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="bfloat16",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model,trust_remote_code=True)
        self.max_seq_length = max_seq_length
        
    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        if "gemma" in self.model_name:
            inputs = [
                {"content": system_prompt + "\n\n" + user_prompt, "role": "user"},
            ]
        else:
            inputs = [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ]
        
        inputs = self.tokenizer.apply_chat_template(inputs, return_tensors="pt", add_special_tokens=True, add_generation_prompt=True).to(self.model.device)
        outputs = self.model.generate(inputs, max_length=self.max_seq_length, do_sample=False, early_stopping=True)
        content = self.tokenizer.decode(outputs[0, inputs.shape[1]:], skip_special_tokens=True)
        return content

class HfDistributedJudge(LLMJudge):
    def __init__(self, model: str, max_seq_length: int = 8192) -> None:
        from accelerate import PartialState
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.distributed_state = PartialState()

        self.model_name = model
        print(f"Loading model {model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="bfloat16",
            device_map="auto"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_seq_length = max_seq_length
        
    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        if "gemma" in self.model_name:
            inputs = [
                {"content": system_prompt + "\n\n" + user_prompt, "role": "user"},
            ]
        else:
            inputs = [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ]
        
        inputs = self.tokenizer.apply_chat_template(inputs, return_tensors="pt", add_special_tokens=True, add_generation_prompt=True).to(self.model.device)
        
        with self.distributed_state.split_between_processes(inputs) as inputs:
            outputs = self.model.generate(inputs, max_length=self.max_seq_length, do_sample=False, early_stopping=True)



class HfInferenceJudge(LLMJudge):
    def __init__(self, model) -> None:
        super().__init__()
        from huggingface_hub import InferenceClient

        token = Path.home() / ".cache" / "huggingface" / "token"
        if not token.exists():
            raise ValueError("Hugging Face token not found")
        with open(token) as f:
            token = f.read().strip()

        self.client = InferenceClient(
            model,
            token=token
        )

    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        message = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}, 
            ],
            max_tokens=2048,
            n=1,
            temperature=0.0,
        )
        print(message)
        return message.choices[0].message.content

class VLLMJudge(LLMJudge):
    def __init__(self, model: str) -> None:
        from vllm import LLM
        self.llm = LLM(model=model, max_model_len=4096, tensor_parallel_size=2)

    @property
    def sampling_params(self) -> 'SamplingParams':
        from vllm import SamplingParams
        return SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens)

    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        if "gemma" in self.model_name:
            inputs = [
                {"content": system_prompt + "\n\n" + user_prompt, "role": "user"},
            ]
        else:
            inputs = [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ]
        
        outputs = self.llm.chat(messages=inputs, sampling_params=self.sampling_params, use_tqdm=False)
        return outputs.outputs[0].text
    
class TogetherAI(LLMJudge):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        import os
        from together import Together

        self.client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
        self.model_name = model_name

    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        if "gemma" in self.model_name:
            inputs = [
                {"content": system_prompt + "\n\n" + user_prompt, "role": "user"},
            ]
        else:
            inputs = [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=inputs,
            temperature=0.0,
            max_tokens=2048,
        )
        
        return response.choices[0].message.content
    
class CohereJudge(LLMJudge):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        import os
        import cohere

        self.client = cohere.ClientV2(api_key=os.environ.get('COHERE_API_KEY'))
        self.model_name = model_name

    @retry.retry(tries=3, delay=60)
    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        if "gemma" in self.model_name:
            inputs = [
                {"content": system_prompt + "\n\n" + user_prompt, "role": "user"},
            ]
        else:
            inputs = [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ]
        
        response = self.client.chat(
            model=self.model_name,
            messages=inputs,
            temperature=0.0,
            max_tokens=2048,
        )
        
        return response.message.content[0].text
    


# response = requests.post(
#     url="https://api.aimlapi.com/chat/completions",
#     headers={
#         "Authorization": "Bearer abc_api_key_xyz",
#         "Content-Type": "application/json",
#     },
#     data=json.dumps(
#         {
#             "model": "gpt-4o",
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": "What kind of model are you?",
#                 },
#             ],
#             "max_tokens": 512,
#             "stream": False,
#         }
#     ),
# )

# response.raise_for_status()
# print(response.json())
class AIMLJudge(LLMJudge):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        import os
        self.api_key = os.environ.get('AIML_API_KEY')
        self.model_name = model_name

    @retry.retry(tries=3, delay=60)
    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            url="https://api.aimlapi.com/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    "max_tokens": 2048,
                    "stream": False,
                    "temperature": 0.0,
                }
            ),
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]