import logging
import operator
from typing import Annotated, TypedDict

from langchain.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from pydantic import Field
from pydantic_settings import BaseSettings

from app.model import LLMSettings, LocalLLMSettings, Model
from app.tool import ToolManager, get_tool_manager_settings_cls

# Initialize logger
logger = logging.getLogger(__name__)


# Define the state structure
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    tool_calls: dict[str, int]


# Define agent settings
class GraphSettings(BaseSettings):
    max_llm_calls: int = Field(
        default=-1,
        description="Maximum number of LLM calls allowed. Use -1 for unlimited.",
    )
    prompt_force_finalize: str = Field(
        default=(
            "Based on the information obtained so far, provide the best possible answer to the user. "
            "You must respond in the same language as the user. "
            "If the information is incomplete, clearly state any uncertainties or limitations in your response."
        ),
        description="Prompt to use after reaching max LLM calls or no available tools are left.",
    )


# Graph class
class Graph:
    def __init__(self, settings: GraphSettings) -> None:
        self._graph_settings = settings
        self._llm_settings = LLMSettings()  # type: ignore
        self._local_llm_settings = LocalLLMSettings()
        self._model = Model(self._llm_settings, self._local_llm_settings)
        tool_manager_settings_cls = get_tool_manager_settings_cls()
        self._tool_manager_settings = tool_manager_settings_cls()
        self._tool_manager = ToolManager(self._tool_manager_settings)
        self._tool_settings = self._tool_manager.get_all_tool_settings()
        self._graph = self._build_graph()

    def _llm_node(self, state: MessagesState) -> MessagesState:
        """LLM decides whether to call a tool or finalize the answer"""

        # LLM call
        logger.info("[LLM Node] LLM call")
        response = (
            self._model.get_model()
            .bind_tools(self._tool_manager.get_available_tools())
            .invoke(state["messages"])
        )
        # Update state
        logger.debug(f"[LLM Node] LLM response: {response}")
        return {
            "messages": [response],
            "llm_calls": state["llm_calls"] + 1,
            "tool_calls": state["tool_calls"],
        }

    def _tool_node(self, state: MessagesState) -> MessagesState:
        """Perform tool calls"""

        if not isinstance(state["messages"][-1], AIMessage):
            raise ValueError("Last message must be an AIMessage to perform tool calls.")
        results: list[AnyMessage] = []
        tool_calls = dict(state["tool_calls"])
        # Perform each tool call
        for tool_call in state["messages"][-1].tool_calls:
            tool_name = tool_call["name"]
            tool = self._tool_manager.get_tool(tool_name)
            if tool:
                logger.info(f"[Tool Node] Tool call: {tool_name}")
                # Invoke tool with error handling
                try:
                    logger.info(f"[Tool Node] Tool args: {tool_call['args']}")
                    result = tool.invoke(tool_call["args"])
                    results.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )
                except Exception as e:
                    logger.warning(
                        f"[Tool Node] Tool execution error: {tool_name}: {str(e)}"
                    )
                    result = f"Tool execution error: {tool_name}: {str(e)}"
                    results.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )
            else:
                # if tool_call is invalid, skip it
                logger.warning(f"[Tool Node] Invalid tool call: {tool_name}")
                results.append(
                    ToolMessage(
                        content=f"Tool error: {tool_name} is invalid or max usage count exceeded.",
                        tool_call_id=tool_call["id"],
                    )
                )
            # Update tool call count
            tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
            # Disable tool if max usage count exceeded
            max_usage_count = self._tool_manager.get_max_usage_count(tool_name)
            if max_usage_count is not None and tool_calls[tool_name] >= max_usage_count:
                self._tool_manager.remove_tool(tool_name)
                logger.info(f"[Tool Node] Max usage count exceeded: {tool_name}")
        # Update state
        logger.debug(f"[Tool Node] Tool results: {results}")
        return {
            "messages": results,
            "llm_calls": state["llm_calls"],
            "tool_calls": tool_calls,
        }

    def _should_use_tools(self, state: MessagesState) -> str:
        """Decide if we should use tools or stop based upon whether the LLM made a tool call"""

        last_message = state["messages"][-1]

        # If the last message is an AIMessage with tool calls, go to tool_node
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_node"
        # Otherwise, go to END
        return END

    def _should_continue(self, state: MessagesState) -> str:
        """Decide if we should continue the loop or stop based upon the number of LLM calls and available tools"""

        # If we have reached the max LLM calls or no tools are left, finalize
        max_llm_calls = self._graph_settings.max_llm_calls
        if (
            max_llm_calls >= 0 and state["llm_calls"] >= max_llm_calls
        ) or not self._tool_manager.get_available_tools():
            return "force_finalize"
        # Otherwise, continue
        return "llm_node"

    def _force_finalize(self, state: MessagesState) -> MessagesState:
        prompt = self._graph_settings.prompt_force_finalize
        # LLM call
        logger.info("[Force Finalize] LLM call")
        response = self._model.get_model().invoke(
            state["messages"] + [SystemMessage(content=prompt)]
        )
        # Update state
        logger.debug(f"[Force Finalize] LLM response: {response}")
        return {
            "messages": [SystemMessage(content=prompt), response],
            "llm_calls": state["llm_calls"] + 1,
            "tool_calls": state["tool_calls"],
        }

    def _build_graph(self):  # type: ignore
        """Build the agent graph"""

        agent_builder = StateGraph(MessagesState)

        agent_builder.add_node("llm_node", self._llm_node)
        agent_builder.add_node("tool_node", self._tool_node)
        agent_builder.add_node("force_finalize", self._force_finalize)

        agent_builder.add_edge(START, "llm_node")
        agent_builder.add_conditional_edges(
            "llm_node",
            self._should_use_tools,
            ["tool_node", END],
        )
        agent_builder.add_conditional_edges(
            "tool_node",
            self._should_continue,
            ["llm_node", "force_finalize"],
        )
        agent_builder.add_edge("force_finalize", END)

        return agent_builder.compile()

    def get_graph(self):  # type: ignore
        """Get the compiled agent graph"""

        return self._graph

    def get_settings(self) -> dict[str, BaseSettings | list[BaseSettings]]:
        """Get current settings"""
        return {
            "graph": self._graph_settings,
            "llm": self._llm_settings,
            "local_llm": self._local_llm_settings,
            "tool_manager": self._tool_manager_settings,
            "tool": self._tool_settings,
        }
