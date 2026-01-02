from typing import Any, Callable, TypedDict

from langchain.tools import BaseTool
from pydantic import Field, create_model
from pydantic.types import PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.tools.current_datetime import (
    CurrentDatetimeSettings,
    create_current_datetime_tool,
    get_current_datetime_tool_name,
)
from app.tools.web_search import (
    WebSearchSettings,
    create_web_search_tools,
    get_web_search_tool_names,
)


# Define type for tool info
class ToolInfo(TypedDict):
    name: str
    enabled: bool
    max_usage_count: int
    tool: BaseTool


# Get tool settings model
def get_tool_manager_settings_cls() -> type[BaseSettings]:
    all_tool_names = [
        *get_web_search_tool_names(),
        *get_current_datetime_tool_name(),
    ]

    # Create dynamic settings model
    fields: dict[str, Any] = {}
    for name in all_tool_names:
        fields["use_" + name.lower()] = (
            bool,
            Field(
                default=True,
                description=f"Whether to enable the {name} tool.",
            ),
        )
        fields["max_usage_count_" + name.lower()] = (
            PositiveInt,
            Field(
                default=1,
                description=f"Maximum number of usage counts for the {name} tool.",
            ),
        )
    return create_model(
        "ToolSettings",
        __base__=BaseSettings,
        __config__=SettingsConfigDict(env_prefix="TOOL_"),
        **fields,
    )


# Define Tool manager class
class ToolManager:
    def __init__(self, settings: BaseSettings) -> None:
        pairs: list[tuple[Callable, type[BaseSettings]]] = [
            (create_current_datetime_tool, CurrentDatetimeSettings),
            (create_web_search_tools, WebSearchSettings),
        ]
        self._tool_settings = [settings_class() for _, settings_class in pairs]
        all_tools = [
            tool
            for (create_func, _), settings in zip(pairs, self._tool_settings)
            for tool in create_func(settings)
        ]
        self._all_tools_info = [
            {
                "name": tool.name,
                "enabled": getattr(settings, "use_" + tool.name.lower()),
                "max_usage_count": getattr(
                    settings, "max_usage_count_" + tool.name.lower()
                ),
                "tool": tool,
            }
            for tool in all_tools
        ]
        self._available_tools: list[BaseTool] = [
            info["tool"] for info in self._all_tools_info if info["enabled"]
        ]

    # Get all tool settings
    def get_all_tool_settings(self) -> list[BaseSettings]:
        return self._tool_settings

    # Get available tools
    def get_available_tools(self) -> list[BaseTool]:
        return [tool for tool in self._available_tools]

    # Check if the tool is available
    def is_available(self, tool_name: str) -> bool:
        for tool in self._available_tools:
            if tool.name == tool_name:
                return True
        return False

    # Get tool by name
    def get_tool(self, tool_name: str) -> BaseTool | None:
        for tool in self._available_tools:
            if tool.name == tool_name:
                return tool
        return None

    # Get max usage count for a tool by name
    def get_max_usage_count(self, tool_name: str) -> int | None:
        for info in self._all_tools_info:
            if info["name"] == tool_name:
                return info["max_usage_count"]
        return None

    # Remove tool by name
    def remove_tool(self, tool_name: str) -> None:
        for tool in self._available_tools:
            if tool.name == tool_name:
                self._available_tools.remove(tool)
                break
