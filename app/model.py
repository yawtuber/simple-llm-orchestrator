from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr
from pydantic.types import NonNegativeInt, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


# Define LLM settings
class LLMSettings(BaseSettings):
    local_hosting: bool = Field(
        default=False,
        description="Whether to use a local LLM model.",
    )
    model_name: str = Field(
        ...,
        description="Name of the LLM model to use.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for the LLM model, if required.",
    )
    retry_attempts: PositiveInt = Field(
        default=3,
        description="Number of retry attempts for LLM model requests.",
    )
    timeout: NonNegativeInt = Field(
        default=60,
        description="Timeout in seconds for LLM model requests.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temperature setting for the LLM model.",
    )

    # prefix for environment variables
    model_config = SettingsConfigDict(env_prefix="LLM_")


# Define local LLM settings
class LocalLLMSettings(BaseSettings):
    protocol: Literal["http", "https"] = Field(
        default="http",
        description="Protocol to use for local LLM model (http or https).",
    )
    host: str = Field(
        default="localhost",
        description="Host address for local LLM model.",
    )
    port: str = Field(
        default="8080",
        description="Port number for local LLM model.",
    )

    # prefix for environment variables
    model_config = SettingsConfigDict(env_prefix="LOCAL_LLM_")


# Define Model class
class Model:
    def __init__(
        self, llm_settings: LLMSettings, local_llm_settings: LocalLLMSettings
    ) -> None:
        base_url: str | None = None
        if llm_settings.local_hosting:
            base_url = (
                f"{local_llm_settings.protocol}://"
                f"{local_llm_settings.host}:{local_llm_settings.port}/v1"
            )
        self.model = ChatOpenAI(
            model=llm_settings.model_name,
            base_url=base_url,
            timeout=llm_settings.timeout,
            max_retries=llm_settings.retry_attempts,
            api_key=llm_settings.api_key,
            temperature=llm_settings.temperature,
        )

    # Get LLM model
    def get_model(self) -> ChatOpenAI:
        return self.model
