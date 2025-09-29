"""
ingest.py
Module for ingesting and processing markdown files from a GitHub repository,
extracting frontmatter metadata, chunking documents, and creating a search index.
"""

import io
import logging
import zipfile

import frontmatter
import requests
from minsearch import Index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_repo_data(repo_owner: str, repo_name: str) -> list[dict]:
    """
    Reads markdown files from a GitHub repository, extracts frontmatter metadata,
    and returns a list of dictionaries containing the metadata and filename.

    Args:
        repo_owner (str): GitHub repository owner.
        repo_name (str): GitHub repository name.

    Returns:
        list[dict]: List of dictionaries with frontmatter metadata and filename.
    """
    prefix = "https://github.com"
    url = f"{prefix}/{repo_owner}/{repo_name}/archive/refs/heads/master.zip"

    logger.info(f"Reading data from repository: {url}")

    resp = requests.get(url)

    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")
    else:
        logger.info("Repository downloaded successfully.")

    repository_data = []

    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename = file_info.filename.lower()

        if not (filename.endswith(".md") and (filename.startswith("ercs-master/ercs"))):
            continue

        with zf.open(file_info) as f_in:
            content = f_in.read().decode("utf-8", errors="ignore")
            post = frontmatter.loads(content)
            data = post.to_dict()

            _, filename_repo = file_info.filename.split("/", maxsplit=1)
            data["filename"] = filename_repo

            repository_data.append(data)

    zf.close()

    logger.info(f"Extracted {len(repository_data)} documents from the repository.")

    return repository_data


def sliding_window(seq: list, size: int, step: int) -> list[dict]:
    """
    Splits a sequence into overlapping chunks using a sliding window approach.
    Args:
        seq (list): The input sequence to be chunked.
        size (int): The size of each chunk.
        step (int): The step size for the sliding window.

    Returns:
        list of dict: A list of dictionaries, each containing the start index and the chunk content.
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        batch = seq[i : i + size]
        result.append({"start": i, "content": batch})
        if i + size > n:
            break

    return result


def chunk_documents(docs, size=2000, step=1000):
    """
    Chunks the "content" field of each document in the list using a sliding window approach.
    Each chunk retains the original document"s metadata.

    Args:
        docs (list of dict): List of documents, each containing a "content" field.
        size (int): The size of each chunk.
        step (int): The step size for the sliding window.

    Returns:
        list of dict: A list of chunked documents with metadata.
    """
    logger.info(f"Chunking {len(docs)} documents with size {size} and step {step}")

    chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop("content")
        doc_chunks = sliding_window(doc_content, size=size, step=step)
        for chunk in doc_chunks:
            chunk.update(doc_copy)
        chunks.extend(doc_chunks)

    logger.info(f"Generated {len(chunks)} chunks from {len(docs)} documents")

    return chunks


def index_data(
    repo_owner: str,
    repo_name: str,
    filter=None,
    chunk=False,
    chunking_params=None,
):
    """
    Reads data from a GitHub repository, optionally filters and chunks the documents,
    and creates a search index.

    Args:
        repo_owner (str): GitHub repository owner.
        repo_name (str): GitHub repository name.
        filter (callable, optional): A function to filter documents. Defaults to None.
        chunk (bool, optional): Whether to chunk documents. Defaults to False.
        chunking_params (dict, optional): Parameters for chunking. Defaults to None.

    Returns:
        Index: A search index created from the documents.
    """
    docs = read_repo_data(repo_owner, repo_name)

    if filter is not None:
        docs = [doc for doc in docs if filter(doc)]

    if chunk:
        if chunking_params is None:
            chunking_params = {"size": 2000, "step": 1000}
        docs = chunk_documents(docs, **chunking_params)

    index = Index(
        text_fields=["content", "filename"],
    )

    index.fit(docs)
    logger.info("Fitting index completed.")

    return index


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    repo_owner = "ethereum"
    repo_name = "ERCs"

    index = index_data(
        repo_owner=repo_owner,
        repo_name=repo_name,
        filter=None,
        chunk=True,
        chunking_params={"size": 2000, "step": 1000},
    )

    print(index)
