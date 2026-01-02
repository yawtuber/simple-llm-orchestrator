import json
import logging
from typing import Literal

from langchain.messages import (
    HumanMessage,
    SystemMessage,
)
from pydantic import Field
from pydantic_settings import BaseSettings

from app.graph import Graph, GraphSettings, MessagesState

# Initialize logger
logger = logging.getLogger(__name__)


# Define app settings
class AppSettings(BaseSettings):
    env: Literal["development", "production"] = Field(
        default="development",
        description="The application environment.",
    )
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"] = (
        Field(
            default="WARNING",
            description="Log level for the application.",
        )
    )


# Load app settings and setup logger configuration
def setup() -> None:
    settings = AppSettings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Current settings for app:")
    logger.info(json.dumps(settings.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    # Initialize graph
    setup()
    graph = Graph(GraphSettings())
    for key, value in graph.get_settings().items():
        logger.info(f"Current settings for {key}:")
        if isinstance(value, list):
            for settings in value:
                logger.info(json.dumps(settings.model_dump(mode="json"), indent=2))
        else:
            logger.info(json.dumps(value.model_dump(mode="json"), indent=2))
    # Example usage
    initial_state: MessagesState = {
        "messages": [
            SystemMessage(
                content="You are a helpful assistant. Provide concise and accurate responses. "
                "When a question requires factual or up-to-date information, "
                "do not rely only on your internal knowledge; use external tools when necessary."
            ),
            HumanMessage(content="Who is the current president of the United States?"),
        ],
        "llm_calls": 0,
        "tool_calls": {},
    }
    logger.info(f"User question: {initial_state['messages'][-1].content}")
    try:
        final_state = graph.get_graph().invoke(initial_state)  # type: ignore
        logger.info(f"LLM answer: {final_state['messages'][-1].content}")
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", stack_info=True)
