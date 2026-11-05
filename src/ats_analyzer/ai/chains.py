"""LangChain Core chains: prompt template -> custom chat model -> JSON repair.

The custom chat model adapter is the LangChain boundary: it standardizes
system/human messages and delegates the actual HTTP work to the injected
AIProvider (which owns the HTTPX plumbing). No HTTP logic is duplicated here
and no provider SDK is involved anywhere.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel, CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ConfigDict

from ats_analyzer.ai import prompts
from ats_analyzer.domain.exceptions import AIOutputError, AtsAnalyzerError
from ats_analyzer.domain.models import AiFeedback, JobRequirements


class ProviderChatModel(BaseChatModel):
    """A minimal LangChain chat model backed by an injected AIProvider.

    Standardized LangChain messages become one provider call, results become
    an AIMessage, and normalized domain exceptions (HTTPClientError,
    ProviderError) propagate unchanged. Unexpected failures become
    AIOutputError with a user-safe message.
    """

    provider: Any  # AIProvider; arbitrary type allowed below
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_prompt = "\n\n".join(
            str(m.content) for m in messages if isinstance(m, SystemMessage)
        )
        user_prompt = "\n\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
        try:
            completion = self.provider.generate(system_prompt=system_prompt, prompt=user_prompt)
        except AtsAnalyzerError:
            raise  # already user-safe and normalized at the provider boundary
        except Exception as exc:  # unexpected adapter failure -> user-safe
            raise AIOutputError(
                f"The AI request failed unexpectedly ({type(exc).__name__})."
            ) from None
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=completion))])

    @property
    def _llm_type(self) -> str:
        return "byok-http-provider"


def build_analysis_chain(provider: Any) -> Runnable:
    """Resume-vs-requirements analysis: template -> provider -> JSON -> AiFeedback."""
    template = ChatPromptTemplate.from_messages(
        [
            # SystemMessage instance -> treated literally (schema contains braces)
            SystemMessage(content=prompts.system_prompt(AiFeedback.model_json_schema())),
            ("human", prompts.load_prompt("resume_analysis")),
        ]
    )
    return (
        template
        | ProviderChatModel(provider=provider)
        | RunnableLambda(prompts.extract_json_object)
        | RunnableLambda(prompts.validate_ai_json(AiFeedback))
    )


def build_extraction_chain(provider: Any) -> Runnable:
    """Job description -> JobRequirements via AI (optional, user-invoked)."""
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompts.system_prompt(JobRequirements.model_json_schema())),
            ("human", prompts.load_prompt("job_extraction")),
        ]
    )
    return (
        template
        | ProviderChatModel(provider=provider)
        | RunnableLambda(prompts.extract_json_object)
        | RunnableLambda(prompts.validate_ai_json(JobRequirements))
    )
