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
    resp = requests.get(url)

    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename = file_info.filename
        filename_lower = filename.lower()

        if not (filename_lower.endswith(".md")):
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


if __name__ == "__main__":
    repo_owner = "your_github_username"
    repo_name = "your_repository_name"
    data = read_repo_data(repo_owner, repo_name)
    logger.info(data)
