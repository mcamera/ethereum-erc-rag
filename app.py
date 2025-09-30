import streamlit as st
import asyncio
from src import ingest, search_agent

# --- Page Configuration ---
st.set_page_config(
    page_title="Ethereum ERCs AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Ethereum ERCs AI Assistant")
st.caption("An AI assistant to answer questions about Ethereum Request for Comments (ERCs)")

# --- Caching the Agent ---
@st.cache_resource
def load_agent(repo_owner, repo_name):
    """
    Initializes the data index and the search agent.
    This will only run once and cache the result.
    """
    # Silent loading - no status messages
    index = ingest.index_data(repo_owner, repo_name, filter=None)
    agent = search_agent.init_agent(index, repo_owner, repo_name)
    return agent

def get_agent_with_status(repo_owner, repo_name):
    """
    Wrapper function to show status only on first load
    """
    # Check if agent is already cached by trying to get it from cache
    cache_key = f"load_agent({repo_owner}, {repo_name})"
    
    # If not in session state, show loading message
    if 'agent_loaded' not in st.session_state:
        with st.status("Initializing AI Assistant (one-time setup)...", expanded=False) as status:
            status.update(label="Downloading and indexing repository...", state="running")
            agent = load_agent(repo_owner, repo_name)
            status.update(label="Ready!", state="complete")
        st.session_state.agent_loaded = True
    else:
        # Silent load for subsequent runs
        agent = load_agent(repo_owner, repo_name)
    
    return agent

def run_async(coro):
    """
    A helper function to run an async coroutine in a sync context (like Streamlit)
    without closing the event loop.
    """
    try:
        # Try to get the running event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If no loop is running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the coroutine until it completes
    return loop.run_until_complete(coro)

async def get_agent_response(agent, user_prompt):
    """Async helper to run the agent and get the response."""
    response = await agent.run(user_prompt=user_prompt)
    return response.output if hasattr(response, 'output') else str(response)

# --- Main Application Logic ---
try:
    REPO_OWNER = "ethereum"
    REPO_NAME = "ERCs"
    
    # This will show status only on first load
    agent = get_agent_with_status(REPO_OWNER, REPO_NAME)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask a question about the {REPO_NAME} repository..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                coroutine = get_agent_response(agent, prompt)
                response_output = run_async(coroutine)
                st.markdown(response_output)
        
        st.session_state.messages.append({"role": "assistant", "content": response_output})

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please try refreshing the page or check the console for more details.")
