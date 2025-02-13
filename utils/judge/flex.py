import openai
import os
from pathlib import Path
from utils import prompts
from .base import OpenAIJudge, JUDGE_PROMPT_STRICT_NORESULT


class FLEX_NoHint(OpenAIJudge):
    USE_HINT = False

    
JUDGE_PROMPT_LOOSE_NORESULT = """
**Schema**
{schema}

**Question**
{question}

**Prediction Query**
{pred_query}

**Ground Truth Query**
{gt_query}
""".strip()

class FLEX_NoResult(OpenAIJudge):

    def get_judge_prompt(self, strict: bool):
        return JUDGE_PROMPT_STRICT_NORESULT if strict else JUDGE_PROMPT_LOOSE_NORESULT

JUDGE_PROMPT_LOOSE_NOQUESTION = """
**Schema**
{schema}

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

JUDGE_PROMPT_STRICT_NOQUESTION = """
**Schema**
{schema}

**Question**
{question}

**Prediction Query**
{pred_query}

**Ground Truth Query**
{gt_query}
""".strip()

class FLEX_NoQuestion(OpenAIJudge):

    def get_judge_prompt(self, strict: bool):
        return JUDGE_PROMPT_STRICT_NOQUESTION if strict else JUDGE_PROMPT_LOOSE_NOQUESTION


JUDGE_PROMPT_LOOSE_NOGT = """
**Schema**
{schema}

**Question**
{question}

**Prediction Query**
{pred_query}

**Prediction Result**
{pred_result}
""".strip()

JUDGE_PROMPT_STRICT_NOGT = """
**Schema**
{schema}

**Question**
{question}

**Prediction Query**
{pred_query}
""".strip()

class FLEX_NoGroundTruth(OpenAIJudge):
    INCLUDE_EX_HINT = False
    
    def get_judge_prompt(self, strict: bool):
        return JUDGE_PROMPT_STRICT_NOGT if strict else JUDGE_PROMPT_LOOSE_NOGT


SYSTEM_STRICT_SIMPLE_NO_CRITERIA = """The Prediction Result matches the Reference Result. However, this does not guarantee that the Prediction Query is correct. Carefully analyze the Prediction Query and evaluate its correctness.
- Both queries are SQLite3 compatible, meaning integer division is not automatically converted to float division, and logical operators, column/table names are case-insensitive.
- comparison between string and integer is allowed in SQLite3.

Analysis Guidelines
1. Compare the Prediction Query with the Ground Truth Query within the context of the provided schema and question.
2. Predict the query's logical correctness based on the criteria mentioned above.

Finally, score the Prediction Query as follows:
```json
{"correct": true or false}
```
""".strip()

SYSTEM_LOOSE_SIMPLE_NO_CRITERIA = """The Prediction Result differs from the Ground Truth Result. However, this does not necessarily mean that the Prediction Query is incorrect. Analyze the differences between the Prediction Query and Reference Query.

- Both queries are SQLite3 compatible, meaning integer division is not automatically converted to float division, and logical operators are case-insensitive.
- If the table schema and description are different, follow the schema provided in the prompt.

Provide a detailed comparison of the Prediction Query and Ground Truth Query, focusing on the nature and significance of their differences. If the Prediction Query is incorrect, explain the specific errors and how they affect the query's ability to answer the Question.

Finally, score the Prediction Query as follows:
```json
{"correct": true/false}
```
""".strip()

class FLEX_NoCriteria(OpenAIJudge):
    def get_system_prompt(self, strict: bool):
        return SYSTEM_STRICT_SIMPLE_NO_CRITERIA if strict else SYSTEM_LOOSE_SIMPLE_NO_CRITERIA
