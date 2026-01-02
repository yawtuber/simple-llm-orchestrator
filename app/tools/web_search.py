from typing import Literal, TypedDict

from ddgs import DDGS
from langchain.tools import BaseTool, tool
from pydantic import Field
from pydantic.types import NonNegativeInt
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import retry, stop_after_attempt, wait_fixed

TOOL_NAME = "web_search"
TOOL_NAME_MINI = "web_search_mini"


def get_web_search_tool_names() -> list[str]:
    return [TOOL_NAME, TOOL_NAME_MINI]


# Load settings from environment variables
class WebSearchSettings(BaseSettings):
    default_region: str = Field(
        default="wt-wt",
        description="Default region for web search.",
    )
    default_safety: Literal["on", "moderate", "off"] = Field(
        default="moderate",
        description="Default safety level for web search. One of 'on', 'moderate', or 'off'.",
    )
    default_timelimit: Literal["d", "w", "m", "y"] | None = Field(
        default=None,
        description=(
            "Default time limit for web search ('d' for day, 'w' for week, 'm' for month, 'y' for year). "
            "Use None for no time limit."
        ),
    )
    default_max_results: NonNegativeInt = Field(
        default=10,
        description="Default maximum number of results to return from web search.",
    )
    default_max_text_length: int = Field(
        default=-1,
        description=(
            "Default maximum text length for web search results. Use -1 for no limit."
        ),
    )
    retry_attempts: NonNegativeInt = Field(
        default=3,
        description="Number of retry attempts for web search.",
    )
    retry_wait_seconds: NonNegativeInt = Field(
        default=2,
        description="Wait time in seconds between retry attempts for web search.",
    )

    # prefix for environment variables
    model_config = SettingsConfigDict(env_prefix="WEB_SEARCH_")


# Define the structure of a web search result
class WebSearchResult(TypedDict):
    title: str
    href: str
    body: str


def result_to_dict(
    result: WebSearchResult,
    max_length: int = -1,
    only_body: bool = False,
) -> dict:
    if only_body:
        body = result["body"]
        if max_length > 0:
            body = body[:max_length] + ("..." if len(body) > max_length else "")
        return {"body": body}
    else:
        title = result["title"]
        href = result["href"]
        body = result["body"]
        if max_length > 0:
            title = title[:max_length] + ("..." if len(title) > max_length else "")
            body = body[:max_length] + ("..." if len(body) > max_length else "")
        return {"title": title, "href": href, "body": body}


# Define custom DDGS with retry
def ddgs_search(
    query: str,
    region: str,
    safesearch: Literal["on", "moderate", "off"],
    timelimit: Literal["d", "w", "m", "y"] | None,
    max_results: int,
    retry_attempts: int,
    retry_wait_seconds: int,
) -> list[WebSearchResult]:
    kwargs = {
        "region": region,
        "safesearch": safesearch,
        "timelimit": timelimit,
        "max_results": max_results,
        "backend": "duckduckgo",
    }

    @retry(
        stop=stop_after_attempt(retry_attempts + 1),
        wait=wait_fixed(retry_wait_seconds),
        reraise=True,
    )
    def ddgs_search_with_retry() -> list[WebSearchResult]:
        results = DDGS().text(query, **kwargs)
        return [
            {"title": r["title"], "href": r["href"], "body": r["body"]} for r in results
        ]

    return ddgs_search_with_retry()


# Define the tool
def create_web_search_tools(
    settings: WebSearchSettings,
) -> list[BaseTool]:
    @tool(TOOL_NAME)
    def web_search(
        query: str,
        region: str,
        safesearch: Literal["on", "moderate", "off"],
        timelimit: Literal["d", "w", "m", "y"] | None,
        max_results: int,
        max_text_length: int,
    ) -> list[dict] | str:
        """
        Search the web for factual information.
        Do not search for information that can be obtained by other tools; use this tool only when necessary.
        Args:
            query (str): The search query.
            region (str): The region code for the search.
                e.g., "us-en" for United States, "jp-jp" for Japan.
            safesearch (Literal["on", "moderate", "off"]): The safesearch level.
            timelimit (Literal["d", "w", "m", "y"] | None): The time limit for the search.
                One of "d" for day, "w" for week, "m" for month, "y" for year.
                should not be specified unless necessary.
            max_results (int): The maximum number of search results to return.
            max_text_length (int): The maximum length of the title and body text in the results.
                The maximum length of the body text in the results.
                Use -1 for no limit.
        """

        results = ddgs_search(
            query,
            region,
            safesearch,
            timelimit,
            max_results,
            settings.retry_attempts,
            settings.retry_wait_seconds,
        )
        results_dict = [result_to_dict(result, max_text_length) for result in results]
        return results_dict

    @tool(TOOL_NAME_MINI)
    def web_search_mini(
        query: str,
    ) -> list[dict] | str:
        """
        Search the web for factual information.
        Do not search for information that can be obtained by other tools; use this tool only when necessary.
        Args:
            query (str): The search query.
        """

        results = ddgs_search(
            query,
            settings.default_region,
            settings.default_safety,
            settings.default_timelimit,
            settings.default_max_results,
            settings.retry_attempts,
            settings.retry_wait_seconds,
        )
        results_dict = [
            result_to_dict(result, settings.default_max_text_length, True)
            for result in results
        ]
        return results_dict

    return [web_search_mini, web_search]
