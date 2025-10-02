import os

os.environ["LOGS_DIRECTORY"] = "eval_logs"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import json

import pandas as pd
from pydantic import BaseModel
from pydantic_ai import Agent
from tqdm.auto import tqdm

from src.logs import LOG_DIR


LOG_DIR.absolute()

evaluation_prompt = """
Use this checklist to evaluate the quality of an AI agent's answer (<ANSWER>) to a user question (<QUESTION>).
We also include the entire log (<LOG> for analysis.

For each item, check if the condition is met. 

Checklist:

- instructions_follow: The agent followed the user's instructions (in <INSRUCTIONS>)
- instructions_avoid: The agent avoided doing things it was told not to do  
- answer_relevant: The response directly addresses the user's question  
- answer_clear: The answer is clear and correct  
- answer_citations: The response includes proper citations or sources when required  
- completeness: The response is complete and covers all key aspects of the request
- tool_call_search: Is the search tool invoked? 

Output true/false for each check and provide a short explanation for your judgment.
"""

user_prompt_format = """
<INSTRUCTIONS>{instructions}</INSTRUCTIONS>
<QUESTION>{question}</QUESTION>
<ANSWER>{answer}</ANSWER>
<LOG>{log}</LOG>
""".strip()


class EvaluationCheck(BaseModel):
    check_name: str
    justification: str
    check_pass: bool


class EvaluationChecklist(BaseModel):
    checklist: list[EvaluationCheck]
    summary: str


eval_agent = Agent(
    name="eval_agent",
    model="gemini-2.5-flash-lite",
    instructions=evaluation_prompt,
    output_type=EvaluationChecklist,
)


def load_log_file(log_file):
    with open(log_file, "r") as f_in:
        log_data = json.load(f_in)
        log_data["log_file"] = log_file
        return log_data


def simplify_log_messages(messages):
    log_simplified = []

    for m in messages:
        parts = []

        for original_part in m["parts"]:
            part = original_part.copy()
            kind = part["part_kind"]

            if kind == "user-prompt":
                del part["timestamp"]
            if kind == "tool-call":
                del part["tool_call_id"]
            if kind == "tool-return":
                del part["tool_call_id"]
                del part["metadata"]
                del part["timestamp"]
            if kind == "tool-return":
                part["content"] = "RETURN_RESULTS_REDACTED"
            if kind == "text":
                del part["id"]

            parts.append(part)

        message = {"kind": m["kind"], "parts": parts}

        log_simplified.append(message)
    return log_simplified


async def evaluate_log_record(eval_agent, log_record):
    messages = log_record["messages"]

    instructions = log_record["system_prompt"]
    question = messages[0]["parts"][0]["content"]
    answer = messages[-1]["parts"][0]["content"]

    log_simplified = simplify_log_messages(messages)
    log = json.dumps(log_simplified)

    user_prompt = user_prompt_format.format(
        instructions=instructions, question=question, answer=answer, log=log
    )

    result = await eval_agent.run(user_prompt, output_type=EvaluationChecklist)
    return result.output


async def run_evaluations():
    """Load evaluation data and run evaluations on log files"""
    eval_set = []

    for log_file in LOG_DIR.glob("*.json"):
        if "gh_agent" not in log_file.name:
            continue

        log_record = load_log_file(log_file)
        if log_record.get("source") != "ai-generated":
            continue

        eval_set.append(log_record)

    print(f"Found {len(eval_set)} log files to evaluate")

    eval_results = []

    for log_record in tqdm(eval_set):
        eval_result = await evaluate_log_record(eval_agent, log_record)
        eval_results.append((log_record, eval_result))

    rows = []

    for log_record, eval_result in eval_results:
        messages = log_record["messages"]

        row = {
            "file": log_record["log_file"].name,
            "question": messages[0]["parts"][0]["content"],
            "answer": messages[-1]["parts"][0]["content"],
        }

        checks = {c.check_name: c.check_pass for c in eval_result.checklist}
        row.update(checks)

        rows.append(row)

    print("Evaluation Summary:")
    df_evals = pd.DataFrame(rows)
    print(df_evals.mean(numeric_only=True) * 100)

    return df_evals


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_evaluations())
