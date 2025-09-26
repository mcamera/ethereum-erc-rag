import io
import logging
import zipfile

import frontmatter
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_repo_data(repo_owner, repo_name):
    """
    Download and parse all markdown files from a GitHub repository.

    Args:
        repo_owner: GitHub username or organization
        repo_name: Repository name

    Returns:
        List of dictionaries containing file content and metadata
    """
    prefix = "https://github.com"
    url = f"{prefix}/{repo_owner}/{repo_name}/archive/refs/heads/master.zip"

    logger.info(f"Downloading repository from {url}")
    resp = requests.get(url)

    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")
    else:
        logger.info("Repository downloaded successfully.")

    repository_data = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename = file_info.filename
        filename_lower = filename.lower()

        if not (
            (filename_lower.endswith(".md"))
            and (filename_lower.startswith("ercs-master/ercs"))
        ):
            continue

        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode("utf-8", errors="ignore")
                post = frontmatter.loads(content)
                data = post.to_dict()
                data["filename"] = filename
                repository_data.append(data)
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    zf.close()

    return repository_data


def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i : i + size]
        result.append({"start": i, "chunk": chunk})
        if i + size >= n:
            break

    return result


def split_markdown_by_level(text, level=2):
    """
    Split markdown text by a specific header level.

    :param text: Markdown text as a string
    :param level: Header level to split on
    :return: List of sections as strings
    """
    import re

    # This regex matches markdown headers
    # For level 2, it matches lines starting with "## "
    header_pattern = r"^(#{" + str(level) + r"} )(.+)$"
    pattern = re.compile(header_pattern, re.MULTILINE)

    # Split and keep the headers
    parts = pattern.split(text)

    sections = []
    for i in range(1, len(parts), 3):
        # We step by 3 because regex.split() with
        # capturing groups returns:
        # [before_match, group1, group2, after_match, ...]
        # here group1 is "## ", group2 is the header text
        header = parts[i] + parts[i + 1]  # "## " + "Title"
        header = header.strip()

        # Get the content after this header
        content = ""
        if i + 2 < len(parts):
            content = parts[i + 2].strip()

        if content:
            section = f"{header}\n\n{content}"
        else:
            section = header
        sections.append(section)

    return sections


def configure_google_ai():
    """
    Configure Google Generative AI with API key from environment variable.
    """
    import google.generativeai as genai
    import os

    from dotenv import load_dotenv

    load_dotenv()

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("The environment variable GOOGLE_API_KEY was not found.")

        genai.configure(api_key=api_key)
        print("✅ Google AI configured successfully!")

    except ValueError as e:
        logger.error(e)

    return genai


def llm(genai, prompt, model="gemini-2.5-flash"):
    """
    Sends a prompt to the Gemini model and returns the response as text.

    Args:
        prompt (str): The text to send to the model.
        model (str): The Gemini model name to use.
                     'gemini-1.5-flash' is a great fast and capable option.

    Returns:
        str: The generated text response from the model.
    """
    model_instance = genai.GenerativeModel(model)

    response = model_instance.generate_content(prompt)

    return response.text


def intelligent_chunking(text):
    prompt = prompt_template.format(document=text)
    response = llm(prompt)
    sections = response.split("---")
    sections = [s.strip() for s in sections if s.strip()]

    return sections


if __name__ == "__main__":
    repo_owner = "ethereum"
    repo_name = "ERCs"
    chunking_method = "sliding_window"  # Options: sliding_window, split_markdown_by_level, intelligent_chunking

    erc_data = read_repo_data(repo_owner, repo_name)
    logger.info(f"The data downloaded contains {len(erc_data)} documents.")

    if chunking_method == "sliding_window":
        # Sliding window example
        logger.info("Using sliding window chunking method.")
        erc_data_chunks = []

        for doc in erc_data:
            doc_copy = doc.copy()
            doc_content = doc_copy.pop("content")
            chunks = sliding_window(doc_content, 2000, 1000)
            for chunk in chunks:
                chunk.update(doc_copy)
            erc_data_chunks.extend(chunks)

        logger.info(
            f"The data after sliding window chunking contains {len(erc_data_chunks)} chunks."
        )

    elif chunking_method == "split_markdown_by_level":
        # Split markdown by level example
        logger.info("Using split markdown by level chunking method.")
        erc_data_chunks = []

        for doc in erc_data:
            doc_copy = doc.copy()
            doc_content = doc_copy.pop("content")
            sections = split_markdown_by_level(doc_content, level=2)
            for section in sections:
                section_doc = doc_copy.copy()
                section_doc["section"] = section
                erc_data_chunks.append(section_doc)

        logger.info(
            f"The data after section splitting contains {len(erc_data_chunks)} section chunks."
        )

    elif chunking_method == "intelligent_chunking":
        # Intelligent Chunking with LLM example
        genai = configure_google_ai()

        prompt_template = """
            Split the provided document into logical sections that make sense for a Q&A system.
            
            Each section should be self-contained and cover a specific topic or concept.
            
            <DOCUMENT>
            {document}
            </DOCUMENT>
            
            Use this format:
            
            ## Section Name
            
            Section content with all relevant details
            
            ---
            
            ## Another Section Name
            
            Another section content
            
            ---
        """.strip()

        from tqdm.auto import tqdm

        erc_data_chunks = []

        for doc in tqdm(erc_data):
            doc_copy = doc.copy()
            doc_content = doc_copy.pop("content")

            sections = intelligent_chunking(doc_content)
            for section in sections:
                section_doc = doc_copy.copy()
                section_doc["section"] = section
                erc_data_chunks.append(section_doc)

        logger.info(
            f"The data after intelligent chunking contains {len(erc_data_chunks)} chunks."
        )

    else:
        raise ValueError(f"Unknown chunking method: {chunking_method}")

    logger.info("Sample chunk:")
    logger.info(erc_data_chunks[0])
