"""OpenAI-compatible chat client (vLLM, Mosaic AI Model Serving, Foundation Model API) as an MLflow LLM span."""
import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from .config import endpoint
from .tools import annotate


class LLM:
    """Callable: (messages, tools, tool_choice) -> assistant message dict.

    Pass `complete=fn(messages, tools, tool_choice) -> (message, usage)` to script the model in tests.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, complete=None):
        self.model = model
        if complete:
            self._complete = complete
        else:
            from openai import OpenAI

            url, key = endpoint()
            self._client = OpenAI(base_url=base_url or url, api_key=api_key or key)
            self._complete = self._openai

    def _openai(self, messages, tools, tool_choice):
        kw = {"tools": tools} if tools else {}
        if tools and tool_choice:
            kw["tool_choice"] = tool_choice
        r = self._client.chat.completions.create(model=self.model, messages=messages, temperature=0, **kw)
        m = r.choices[0].message
        msg = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [
                {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}}
                for t in m.tool_calls
            ]
        u = r.usage
        return msg, {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {}

    @mlflow.trace(name="llm_call", span_type=SpanType.LLM)
    def __call__(self, messages: list, tools: list | None = None, tool_choice: dict | None = None) -> dict:
        msg, usage = self._complete(messages, tools, tool_choice)
        annotate(model=self.model, **({SpanAttributeKey.CHAT_USAGE: usage} if usage else {}))
        return msg
