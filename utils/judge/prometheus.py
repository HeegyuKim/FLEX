from utils.judge.base import OpenAIJudge, HfJudge, DB_HINTS
import re



ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback: """

SCORE_RUBRIC_TEMPLATE = """
[{criteria}]
Score 1: {score1_description}
Score 2: {score2_description}
Score 3: {score3_description}
Score 4: {score4_description}
Score 5: {score5_description}
""".strip()

RUBRIC_AMBIG = {
    "criteria": "How accurately does the model translate natural language queries into correct and efficient SQL queries?",
    "score1_description": "The generated SQL query is completely incorrect and does not address the user's request.",
    "score2_description": "The SQL query partially addresses the user's request but contains significant errors or omissions.",
    "score3_description": "The SQL query mostly addresses the user's request with minor errors or inefficiencies.",
    "score4_description": "The SQL query correctly addresses the user's request with only trivial issues, if any.",
    "score5_description": "The SQL query perfectly translates the user's request into an accurate and efficient SQL statement."
}

RUBRIC_FP = {
  "criteria": "How accurately does the Prediction Query answer the Question and adhere to SQLite3 compatibility?",
  "score1_description": "The Prediction Query is entirely incorrect, failing to answer the Question logically or containing critical errors that render it unusable. It may also be incompatible with SQLite3.",
  "score2_description": "The Prediction Query has significant flaws, such as missing essential tables/columns, incorrect filtering, or JOIN redundancy. It may partially answer the Question but has major issues. SQLite3 compatibility might be questionable.",
  "score3_description": "The Prediction Query somewhat answers the Question but has noticeable issues. It may fail to handle null values properly, misuse aggregation functions, or not consider multiple row scenarios. It is mostly SQLite3 compatible but may have minor issues.",
  "score4_description": "The Prediction Query largely answers the Question correctly with only minor flaws. It handles most scenarios well, including null values and multiple rows. It is fully SQLite3 compatible. Small oversights like missing DISTINCT in SELECT are acceptable.",
  "score5_description": "The Prediction Query perfectly answers the Question, considering all aspects including proper handling of null values, multiple rows, and edge cases. It is fully SQLite3 compatible and optimized. Minor differences like missing DISTINCT in SELECT are allowed if they don't affect the result."
}

RUBRIC_FN = {
  "criteria": "How accurately does the Prediction Query answer the Question compared to the Ground Truth Query, considering logical correctness and result similarity?",
  "score1_description": "The Prediction Query completely fails to answer the Question logically. It contains significant errors, produces drastically different results from the Ground Truth Query, or fails to handle critical aspects like null values or multiple rows.",
  "score2_description": "The Prediction Query partially answers the Question but has major flaws. It may have incorrect filtering, missing conditions, or JOIN redundancy that significantly affect the results. The output differs notably from the Ground Truth Query in ways that impact the answer's accuracy.",
  "score3_description": "The Prediction Query generally answers the Question, but with some inconsistencies. It may have minor issues in handling null values or multiple rows. The results differ from the Ground Truth Query in ways that slightly affect the answer's completeness or accuracy.",
  "score4_description": "The Prediction Query correctly answers the Question with only minor differences from the Ground Truth Query. It handles null values and multiple rows appropriately. Any differences in output structure or value representation do not significantly impact the answer's accuracy.",
  "score5_description": "The Prediction Query perfectly answers the Question, either matching the Ground Truth Query or providing an equally valid alternative. It handles all aspects correctly, including null values and multiple rows. Any differences in output structure, column naming, or value representation are logically justified and do not affect the answer's accuracy."
}

def parse_output_absolute(output):
    pattern = r"""
        (?:                        # Start of non-capturing group
            \[RESULT\]|\[SCORE\]|   # Match [RESULT] or [SCORE]
            Score:?|score:?|        # Match Score: or score:
            Result:?|\[Result\]:?|  # Match Result: or [Result]:
            score\s+of              # Match "score of"
        )                           # End of non-capturing group
        \s*                         # Allow any whitespace
        (?:\(|\[|\s)*               # Allow opening brackets or whitespace
        (\d+)                       # Capture the digit(s)
        (?:                         # Start of non-capturing group
            (?:\)|\]|\s|$)|         # Allow closing brackets, whitespace, or end of string
            (?:/\s*5|               # Allow /5 with optional whitespace
               \s*out\s*of\s*5)     # or "out of 5" with flexible whitespace
        )?                          # End of non-capturing group
        (?:\s*$)                    # Match from the end of string 
    """
    match = re.search(pattern, output, re.IGNORECASE | re.VERBOSE)

    if match:
        result = int(match.group(1))
        if 1 <= result <= 5:  # Ensure the result is within the valid range
            feedback = output[: match.start()].strip()
            return output, result

    return None, None

class PrometheusJudge(HfJudge):

    def __init__(self, max_seq_length: int = 8192) -> None:
        super().__init__("prometheus-eval/prometheus-7b-v2.0", max_seq_length)

    def judge(self, schema: str, question: str, pred_query, pred_result: str, gt_query, gt_result: str, hint: str = None, verbose: bool = False, strict: bool = True) -> str:
        
        if strict:
            rubric = SCORE_RUBRIC_TEMPLATE.format(**RUBRIC_FP)
        else:
            rubric = SCORE_RUBRIC_TEMPLATE.format(**RUBRIC_FN)

        prompt = f"Answer the questiong using following DB schema:\n{schema}\n\nQuestion: {question}"
        prompt += f"\n\nNote: two queries have {'same' if strict else 'different'} execution results.\n"
        prompt += "\n**Hint**\n" + DB_HINTS + (hint or "")


        user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(
            instruction=prompt,
            response=pred_query + "\n\nResult:\n" + pred_result,
            reference_answer=gt_query + "\n\nResult:\n" + gt_result,
            rubric=rubric,   
        )
        
        content = self.request_llm(ABS_SYSTEM_PROMPT, user_content)
        feedback, result = parse_output_absolute(content)

        if verbose:
            print(user_content)
            print("-->", content)
            print("---\n")
        return feedback, {"score": result}

    def request_llm(self, system_prompt: str, user_prompt: str) -> str:
        inputs = [
            {"content": user_prompt, "role": "user"},
            ]
        
        inputs = self.tokenizer.apply_chat_template(inputs, return_tensors="pt", add_special_tokens=True, add_generation_prompt=True).to(self.model.device)
        outputs = self.model.generate(inputs, max_length=self.max_seq_length, do_sample=False, early_stopping=True)
        content = self.tokenizer.decode(outputs[0, inputs.shape[1]:], skip_special_tokens=True)
        return content


class PrometheusAmbiguousJudge(PrometheusJudge):

    def judge(self, schema: str, question: str, pred_query, pred_result: str, gt_query, gt_result: str, hint: str = None, verbose: bool = False, strict: bool = True) -> str:
        rubric = SCORE_RUBRIC_TEMPLATE.format(**RUBRIC_AMBIG)
        prompt = f"Answer the questiong using following DB schema:\n{schema}\n\nQuestion: {question}"
        prompt += f"\n\nNote: two queries have {'same' if strict else 'different'} execution results.\n"
        prompt += "\n**Hint**\n" + DB_HINTS + (hint or "")


        user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(
            instruction=prompt,
            response=pred_query + "\n\nResult:\n" + pred_result,
            reference_answer=gt_query + "\n\nResult:\n" + gt_result,
            rubric=rubric,   
        )
        
        content = self.request_llm(ABS_SYSTEM_PROMPT, user_content)
        feedback, result = parse_output_absolute(content)

        if verbose:
            print(user_content)
            print("-->", content)
            print("---\n")
        return feedback, {"score": result}
    

class PrometheusOpenAIJudge(OpenAIJudge):
    
    def judge(self, schema: str, question: str, pred_query, pred_result: str, gt_query, gt_result: str, hint: str = None, verbose: bool = False, strict: bool = True) -> str:
        
        if strict:
            rubric = SCORE_RUBRIC_TEMPLATE.format(**RUBRIC_FP)
        else:
            rubric = SCORE_RUBRIC_TEMPLATE.format(**RUBRIC_FN)

        prompt = f"Answer the questiong using following DB schema:\n{schema}\n\nQuestion: {question}"
        prompt += f"\n\nNote: two queries have {'same' if strict else 'different'} execution results.\n"
        prompt += "\n**Hint**\n" + DB_HINTS + (hint or "")


        user_content = ABSOLUTE_PROMPT.format(
            instruction=prompt,
            response=pred_query + "\n\nResult:\n" + pred_result,
            reference_answer=gt_query + "\n\nResult:\n" + gt_result,
            rubric=rubric,   
        )
        
        content = self.request_llm(ABS_SYSTEM_PROMPT, user_content)
        feedback, result = parse_output_absolute(content)

        if verbose:
            print(user_content)
            print("-->", content)
            print("---\n")
        return feedback, {"score": result}


class PrometheusAmbiguousOpenAIJudge(OpenAIJudge):

    def judge(self, schema: str, question: str, pred_query, pred_result: str, gt_query, gt_result: str, hint: str = None, verbose: bool = False, strict: bool = True) -> str:
        rubric = SCORE_RUBRIC_TEMPLATE.format(**RUBRIC_AMBIG)
        prompt = f"Answer the questiong using following DB schema:\n{schema}\n\nQuestion: {question}"
        prompt += f"\n\nNote: two queries have {'same' if strict else 'different'} execution results.\n"
        prompt += "\n**Hint**\n" + DB_HINTS + (hint or "")


        user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(
            instruction=prompt,
            response=pred_query + "\n\nResult:\n" + pred_result,
            reference_answer=gt_query + "\n\nResult:\n" + gt_result,
            rubric=rubric,   
        )
        
        content = self.request_llm(ABS_SYSTEM_PROMPT, user_content)
        feedback, result = parse_output_absolute(content)

        if verbose:
            print(user_content)
            print("-->", content)
            print("---\n")
        return feedback, {"score": result}
    