from typing import List, Tuple
import os
import re
import google.generativeai as genai

from .judge import OpenAIJudge, HfJudge, PrometheusJudge, LLMSQLSolver, MistralAIJudge, GeminiJudge, AnthropicJudge, HfInferenceJudge, FLEX_NoResult, FLEX_NoQuestion, FLEX_NoHint, FLEX_NoCriteria, FLEX_NoGroundTruth, PrometheusAmbiguousJudge, PrometheusOpenAIJudge, PrometheusAmbiguousOpenAIJudge, TogetherAI, CohereJudge, AIMLJudge


def get_model(model: str, base_url: str = None):
    if model.startswith("gpt") or model.startswith("o1"):
        return OpenAIJudge(model, base_url), model
    if model in ["deepseek-chat", "deepseek-coder"]:
        return OpenAIJudge(model, "https://api.deepseek.com/v1", os.environ["DEEPSEEK_API_KEY"]), model

    elif model.startswith("ablation/"):
        model = re.sub(r"ablation/", "", model)
        ablation_type, model = model.split("/", 1)
        name = f"{model}-{ablation_type}"
        
        if ablation_type == "no-result":
            return FLEX_NoResult(model, base_url), name
        elif ablation_type == "no-question":
            return FLEX_NoQuestion(model, base_url), name
        elif ablation_type == "no-hint":
            return FLEX_NoHint(model, base_url), name
        elif ablation_type == "no-criteria":
            return FLEX_NoCriteria(model, base_url), name
        elif ablation_type == "no-gt":
            return FLEX_NoGroundTruth(model, base_url), name
        else:
            raise ValueError(f"Invalid ablation study name: {ablation_type}")

    
    elif model.startswith("gemini"):
        return GeminiJudge(model), model
    
    elif model.startswith("claude"):
        return AnthropicJudge(model), model
    
    elif model.startswith("mistral-api/"):
        model = re.sub(r"mistral-api/", "", model)
        return MistralAIJudge(model), model

    elif model.startswith("vllm-server/"):
        model = re.sub(r"vllm-server/", "", model)
        return OpenAIJudge(model, "http://localhost:8000/v1", "token-abc123"), model.replace("/", "-")
    
    elif model.startswith("hf-inference-api"):
        model = model.replace("hf-inference-api/", "")
        return HfInferenceJudge(model), model.replace("/", "-")
    
    elif model.startswith("hf"):
        model = re.sub(r"hf/", "", model)
        return HfJudge(model), model.replace("/", "-")
    
    elif model.startswith("togetherai/"):
        model = re.sub(r"togetherai/", "", model)
        return TogetherAI(model), model.replace("/", "-")
    
    elif model.startswith("cohere/"):
        model = re.sub(r"cohere/", "", model)
        return CohereJudge(model), model.replace("/", "-")
    
    elif model.startswith("aiml/"):
        model = re.sub(r"aiml/", "", model)
        # return OpenAIJudge(model, base_url="https://api.aimlapi.com", api_key=os.environ["AIML_API_KEY"]), model.replace("/", "-")
        return AIMLJudge(model), model.replace("/", "-")
    
    elif model == "prometheus-customized":
        return PrometheusJudge(), model
    elif model == "prometheus-ambiguous":
        return PrometheusAmbiguousJudge(), model
    elif model.startswith("prometheus-ambiguous-openai/"):
        model = re.sub(r"prometheus-ambiguous-openai/", "", model)
        return PrometheusAmbiguousOpenAIJudge(model), "prometheus-ambiguous-" + model
    elif model.startswith("prometheus-openai/"):
        model = re.sub(r"prometheus-openai/", "", model)
        return PrometheusOpenAIJudge(model), "prometheus-" + model
    
    elif model.startswith("llm-sql-solver/"):
        model_name = re.sub(r"llm-sql-solver/", "", model)
        return LLMSQLSolver(model_name), model.replace("/", "-")
    
    else:
        raise ValueError(f"Invalid model: {model}")
    

if __name__ == "__main__":
    judge, _ = get_model("hf-inference-api/mistralai/Mixtral-8x7B-Instruct-v0.1")

    schema = "CREATE TABLE schools (County TEXT, FundingType TEXT, DOC TEXT, DOCType TEXT);"
    question = "List the counties and funding types of schools."
    pred_query = "SELECT County, FundingType FROM schools;"
    pred_result = "ALAMEDA, Public; CONTRA COSTA, Charter"
    gt_query = "SELECT County, FundingType FROM schools;"
    gt_result = "ALAMEDA, Public; CONTRA COSTA, Charter"
    hint = "Hint: schools.FundingType and frpm.`Charter Funding Type` are not equivalent"
    result = judge.judge(schema, question, pred_query, pred_result, gt_query, gt_result, hint, verbose=True, strict=True)
    print(result)
