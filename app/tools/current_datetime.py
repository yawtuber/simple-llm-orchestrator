import datetime

from langchain.tools import BaseTool, tool
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

TOOL_NAME = "current_datetime"


# Get tool name
def get_current_datetime_tool_name() -> list[str]:
    return [TOOL_NAME]


# Define settings for the tool
class CurrentDatetimeSettings(BaseSettings):
    # Default format: "2026-01-01 00:00:00 Thursday"
    format: str = Field(
        default="%Y-%m-%d %H:%M:%S %A",
        description="The format string for the current date and time.",
    )

    # prefix for environment variables
    model_config = SettingsConfigDict(env_prefix="CURRENT_DATETIME_")


# Define the tool
def create_current_datetime_tool(
    settings: CurrentDatetimeSettings,
) -> list[BaseTool]:
    @tool(TOOL_NAME)
    def current_datetime() -> str:
        """The single source of truth for the current date and time."""

        now = datetime.datetime.now()
        return now.strftime(settings.format)

    return [current_datetime]
