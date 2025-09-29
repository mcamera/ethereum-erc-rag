"""
Module to initialize and configure the search agent.
"""

from pydantic_ai import Agent

import search_tools

SYSTEM_PROMPT_TEMPLATE = """
    You are an expert in Computer Science and Distributed Ledger Technology, with an emphasis on the Ethereum blockchain.

    Use the search tool to find relevant information from the Ethereum ERC materials before answering questions.  

    If you can find specific information through search, use it to provide accurate answers.

    Always include references by citing the filename of the source material you used.  
    Replace only the final references with the full path to the GitHub repository:
    "https://github.com/{repo_owner}/{repo_name}/blob/master/"
    Format: [LINK TITLE](FULL_GITHUB_LINK)
    Don't replace any other links of the document.

    If the search doesn't return relevant results, let the user know and provide general guidance.
 """


def init_agent(index: search_tools.Index, repo_owner: str, repo_name: str) -> Agent:
    """
    Initialize the search agent with the given index and repository information.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        repo_owner=repo_owner, repo_name=repo_name
    )

    search_tool = search_tools.SearchTool(index=index)

    agent = Agent(
        name="gh_agent",
        instructions=system_prompt,
        tools=[search_tool.search],
        model="gemini-2.5-flash",
    )

    return agent
