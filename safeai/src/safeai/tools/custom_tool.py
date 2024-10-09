from crewai_tools import BaseTool
from requests import get


class GithubSearch(BaseTool):
    """_summary_ Takes a Github Repo URL and returns
    - Search for a file in the repo
    - Search for a method in the repo, file A tool to search Github Responses"""
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, you agent will need this information to use it."
    )

    def _run(self, argument: url) -> str:
        # Implementation goes here
        return get(url)
