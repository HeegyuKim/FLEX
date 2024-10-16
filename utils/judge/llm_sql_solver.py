from .base import OpenAIJudge
from utils import prompts

class LLMSQLSolver(OpenAIJudge):

    def __init__(self, model: str, base_url: str = None, api_key: str = None) -> None:
        super().__init__(model, base_url, api_key)
        self.prompt_format = prompts.LLM_SQL_SOLVER_CounterExample
        self.max_tokens = 3000

    def judge(self, schema: str, question: str, pred_query, pred_result: str, gt_query, gt_result: str, hint: str = None, verbose: bool = False, strict: bool = True) -> str:
        
        prompt_format = prompts.LLM_SQL_SOLVER_MiniatureAndMull if strict else prompts.LLM_SQL_SOLVER_ExplainAndCompare

        prompt = prompt_format.format(
            schema=schema,
            Q1=pred_query,
            Q2=gt_query,
        )

        content = self.request_llm(None, prompt)

        if verbose:
            print(prompt)
            print("-->", content)
            print("---\n")

        if "NOT EQUIVALENT" in content:
            return content, False
        elif "EQUIVALENT" in content:
            return content, True
        else:
            return content, "error"
