import json
import os
import random
import time

# Set the logs directory BEFORE importing anything that uses it
os.environ["LOGS_DIRECTORY"] = "eval_logs"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from pydantic_ai import Agent
from tqdm.auto import tqdm

from src.logs import LOG_DIR, log_interaction_to_file
from src.main import initialize_agent, initialize_index


QUESTION_GENERATION_PROMPT = """
You are helping to create test questions for an AI agent that answers questions about the Ethereum ERC documentation.

Based on the provided data content, generate realistic questions that users might ask.

The questions should:

- Be natural and varied in style
- Range from simple to complex
- Include both specific technical questions and general course questions

Generate one question for each record.
""".strip()

LOG_DIR.absolute()
index = initialize_index()
agent = initialize_agent(index)


class QuestionsList(BaseModel):
    questions: list[str]


question_generator = Agent(
    name="question_generator",
    instructions=QUESTION_GENERATION_PROMPT,
    model="gemini-2.5-flash",
    output_type=QuestionsList,
)


async def generate_questions_and_answers():
    """Generate 10 questions from sample documents and get answers from the agent"""
    sample = random.sample(index.docs, 10)
    prompt_docs = [d["content"] for d in sample]
    prompt = json.dumps(prompt_docs)

    result = await question_generator.run(prompt)
    questions = result.output.questions
    print(f"Generated {len(questions)} questions")

    for q in tqdm(questions):
        print(f"Question: {q}")

        result = await agent.run(user_prompt=q)
        print(f"Answer: {result.output}")

        log_interaction_to_file(agent, result.new_messages(), source="ai-generated")
        time.sleep(5)
        print()


if __name__ == "__main__":
    import asyncio

    asyncio.run(generate_questions_and_answers())
