
# export OPENAI_API_KEY=

eval() {
    model=$1
    name=$2

    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fn.jsonl $model
    python llm_human_agreement.py text2sql-baselines/human_eval/0807/fp.jsonl $model
}

if [ -z "$1" ]; then
    echo "Please provide the model name"
    exit 1
fi

if [ $1 == "gemini" ]; then
    # eval "gemini-1.5-pro"
    # eval "gemini-1.5-flash"
    eval "gemini-1.0-pro"
fi
# eval "gpt-4o-mini"
if [ $1 == "claude" ]; then
    export ANTHROPIC_API_KEY=YOUR_API_KEY
    # eval "claude-3-sonnet-2024029"
    eval "claude-3-opus-20240229"
    # eval "claude-3-haiku-20240307"
    # eval "claude-3-5-sonnet-20240620"
fi

if [ $1 == "gpt" ]; then
    # eval "gpt-4o-2024-08-06"
    # eval "gpt-4o-2024-05-13"
    # eval "gpt-4o-2024-05-13"
    # eval "gpt-4-1106-preview"
    # eval "o1-mini"
    # eval "o1-preview"
    # eval "gpt-4-turbo-2024-04-09"
    # eval "gpt-4-0125-preview"
    eval "gpt-4o-mini-2024-07-18"
fi

if [ $1 == "ablation" ]; then

    eval "ablation/no-result/gpt-4o-2024-08-06"
    eval "ablation/no-question/gpt-4o-2024-08-06"
    eval "ablation/no-hint/gpt-4o-2024-08-06"
    eval "ablation/no-criteria/gpt-4o-2024-08-06"
    eval "ablation/no-gt/gpt-4o-2024-08-06"
fi

if [ $1 == "mistral" ]; then
    eval "mistral-api/mistral-large-2407"
    eval "mistral-api/mistral-small-2409"
    eval "mistral-api/codestral-2405"
fi

if [ $1 == "deepseek" ]; then
    eval "deepseek-chat"
    eval "deepseek-coder"
fi