"""
OpenAI and Gemini model calls for agent replies.
Includes optional Tavily web search. Uses a callback for message building to avoid depending on the app.
"""

import json
import os
import time
from typing import Callable, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

# Optional Tavily client for web search (TAVILY_API_KEY in env)
_tavily_client = None
_tavily_error: Optional[str] = None


def get_tavily_client():
    """Return Tavily client if TAVILY_API_KEY is set and client creation succeeded; else None."""
    global _tavily_client, _tavily_error
    if _tavily_client is None and os.environ.get("TAVILY_API_KEY"):
        try:
            from tavily import TavilyClient
            _tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            _tavily_error = None
        except Exception as e:
            _tavily_error = str(e)
    return _tavily_client


def get_tavily_error() -> Optional[str]:
    """If Tavily client creation failed, return the error message; else None."""
    get_tavily_client()  # ensure we've tried to create the client
    return _tavily_error


def get_tavily_status() -> str:
    """Return the Tavily (web search) status caption for the UI."""
    client = get_tavily_client()
    key_set = bool(os.environ.get("TAVILY_API_KEY"))
    err = get_tavily_error()
    if client:
        return "Tavily (web search): enabled — agents can look up current information."
    if key_set and err:
        return f"Tavily (web search): disabled — key set but client failed: {err}"
    if key_set:
        return "Tavily (web search): disabled — key set but client failed. Check that tavily-python is installed."
    return "Tavily (web search): disabled — TAVILY_API_KEY not in environment. Add it to .env next to app.py (or run from project root)."


def _run_tavily_search(query: str) -> str:
    """Run a Tavily search and return a summary string for the model."""
    client = get_tavily_client()
    if not client:
        return "Web search is not available (TAVILY_API_KEY not set)."
    try:
        response = client.search(query=query, max_results=5, search_depth="basic")
        results = response.get("results", [])
        if not results:
            return "No results found for that query."
        parts = []
        for i, r in enumerate(results[:5], 1):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            parts.append(f"[{i}] {title}\nURL: {url}\n{content}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Search failed: {e}"


def _run_tool(name: str, args: dict) -> str:
    """Execute a tool by name (e.g. web_search). Used internally by model calls."""
    if name == "web_search":
        return _run_tavily_search(args.get("query", ""))
    return f"Unknown tool: {name}"


# OpenAI tool definition for web search (used by both OpenAI and Gemini)
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for up-to-date information. ALWAYS use this for: questions about recent or future events, current facts, dates after your knowledge cutoff, or anything you are unsure about. Do not answer such questions from memory—call web_search first, then answer using the results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to run"},
            },
            "required": ["query"],
        },
    },
}


def _get_openai_model() -> str:
    """Model from OPENAI_MODEL env; default gpt-5-mini."""
    return (os.environ.get("OPENAI_MODEL") or "").strip() or "gpt-5-mini"


def _get_openai_temperature(model: str) -> Optional[float]:
    """Temperature for chat completion: only when model is gpt-4o-mini, from OPENAI_TEMPERATURE; else None (API default)."""
    if model != "gpt-4o-mini":
        return None
    s = (os.environ.get("OPENAI_TEMPERATURE") or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _get_openai_chat_kwargs(messages: list, *, tools: Optional[list] = None, stream: bool = False) -> dict:
    """Build kwargs for OpenAI chat.completions.create: model, messages, stream, optional tools and temperature."""
    model = _get_openai_model()
    kwargs = {"model": model, "messages": messages, "stream": stream}
    if tools:
        kwargs["tools"] = tools
    temp = _get_openai_temperature(model)
    if temp is not None:
        kwargs["temperature"] = temp
    return kwargs


def _use_gemini() -> bool:
    """True if USE_MODEL env is 'gemini' and Gemini client is available."""
    if genai is None or genai_types is None:
        return False
    return (os.environ.get("USE_MODEL") or "").strip().lower() == "gemini"


def _get_gemini_client():
    """Gemini client using GEMINI_API_KEY. Call only when _use_gemini() is True."""
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _get_gemini_model() -> str:
    """Model from GEMINI_MODEL env; default gemini-2.0-flash."""
    return (os.environ.get("GEMINI_MODEL") or "").strip() or "gemini-2.0-flash"


def get_model_status() -> str:
    """Return the model (reasoning) status for the UI, e.g. 'Model: OpenAI / gpt-5-mini' or 'Model: Gemini / gemini-2.0-flash'."""
    if _use_gemini():
        return f"Model: Gemini / {_get_gemini_model()}"
    return f"Model: OpenAI / {_get_openai_model()}"


def _openai_messages_to_gemini_contents(messages: list) -> tuple[list, Optional[str]]:
    """Convert OpenAI-format messages to (Gemini contents list, system_instruction or None).
    Drops system from contents and returns it as second element for config.system_instruction."""
    system_parts = []
    contents = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        if role == "user":
            contents.append(genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=content)]))
            continue
        if role == "assistant":
            parts = []
            if content:
                parts.append(genai_types.Part.from_text(text=content))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args_str = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                parts.append(genai_types.Part.from_function_call(name=name, args=args))
            if parts:
                contents.append(genai_types.Content(role="model", parts=parts))
            continue
        if role == "tool":
            name = "web_search"
            contents.append(genai_types.Content(
                role="tool",
                parts=[genai_types.Part.from_function_response(name=name, response={"result": m.get("content") or ""})],
            ))
            continue
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return contents, system_instruction


def _gemini_search_tool():
    """Gemini Tool for web_search (single tool)."""
    return genai_types.Tool(function_declarations=[
        genai_types.FunctionDeclaration(
            name="web_search",
            description=SEARCH_TOOL["function"]["description"],
            parameters_json_schema=SEARCH_TOOL["function"]["parameters"],
        )
    ])


def _stream_chat_completion(client, messages: list, tools: Optional[list], placeholder) -> str:
    """Run a streaming chat completion; update placeholder with accumulated text. Returns full reply text."""
    kwargs = _get_openai_chat_kwargs(messages, tools=tools, stream=True)
    stream = client.chat.completions.create(**kwargs)
    accumulated = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None) or (delta.get("content") if isinstance(delta, dict) else None)
        if content:
            accumulated += content
            placeholder.text(accumulated)
    return accumulated.strip()


def _stream_gemini(client, contents: list, system_instruction: Optional[str], tools: Optional[list], placeholder) -> str:
    """Run streaming Gemini generate_content; update placeholder with accumulated text. Returns full reply text."""
    model = _get_gemini_model()
    config = genai_types.GenerateContentConfig(system_instruction=system_instruction or "")
    if tools:
        config.tools = tools
    accumulated = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if chunk.text:
            accumulated += chunk.text
            placeholder.text(accumulated)
    return accumulated.strip()


def call_openai_for_agent(
    role_prompt: str,
    speaker: str,
    stream_placeholder=None,
    *,
    build_messages_for_agent: Callable[[str, str, Optional[str]], list],
) -> tuple[str, list]:
    """Call OpenAI chat completion for the given agent. Returns (reply_text, messages_sent)."""
    if OpenAI is None:
        raise ImportError("The openai package is required when USE_MODEL is not 'gemini'. Install with: pip install openai")
    role_text_only = None
    messages = build_messages_for_agent(role_prompt, speaker, role_text_only)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    tools = [SEARCH_TOOL] if get_tavily_client() else None
    max_tool_rounds = 5
    for _ in range(max_tool_rounds):
        if not tools and stream_placeholder:
            reply = _stream_chat_completion(client, messages, None, stream_placeholder)
            return (reply, messages)
        kwargs = _get_openai_chat_kwargs(messages, tools=tools, stream=False)
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        if not getattr(msg, "tool_calls", None):
            reply = (msg.content or "").strip()
            if stream_placeholder and reply:
                for i in range(1, len(reply) + 1):
                    stream_placeholder.text(reply[:i])
                    time.sleep(0.01)
            return (reply, messages)
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            result = _run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    if stream_placeholder:
        reply = _stream_chat_completion(client, messages, None, stream_placeholder)
    else:
        kwargs = _get_openai_chat_kwargs(messages, stream=False)
        final = client.chat.completions.create(**kwargs)
        reply = (final.choices[0].message.content or "").strip()
    return (reply, messages)


def call_gemini_for_agent(
    role_prompt: str,
    speaker: str,
    stream_placeholder=None,
    *,
    build_messages_for_agent: Callable[[str, str, Optional[str]], list],
) -> tuple[str, list]:
    """Call Gemini for the given agent. Returns (reply_text, messages_sent)."""
    role_text_only = None
    messages = build_messages_for_agent(role_prompt, speaker, role_text_only)
    client = _get_gemini_client()
    gemini_tools = [_gemini_search_tool()] if get_tavily_client() else None
    max_tool_rounds = 5
    for _ in range(max_tool_rounds):
        contents, system_instruction = _openai_messages_to_gemini_contents(messages)
        if not gemini_tools and stream_placeholder:
            reply = _stream_gemini(client, contents, system_instruction, None, stream_placeholder)
            return (reply, messages)
        config = genai_types.GenerateContentConfig(system_instruction=system_instruction or "")
        if gemini_tools:
            config.tools = gemini_tools
        response = client.models.generate_content(model=_get_gemini_model(), contents=contents, config=config)
        text = (response.text or "").strip()
        function_calls = getattr(response, "function_calls", None) or []
        if not function_calls and getattr(response, "candidates", None):
            cand = response.candidates[0] if response.candidates else None
            if cand and getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                for part in cand.content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        function_calls.append(fc)
        if not function_calls:
            reply = text
            if stream_placeholder and reply:
                for i in range(1, len(reply) + 1):
                    stream_placeholder.text(reply[:i])
                    time.sleep(0.01)
            return (reply, messages)

        def _fc_name_args(fc):
            name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else "web_search")
            args = getattr(fc, "args", None)
            if args is None and hasattr(fc, "function_call"):
                args = getattr(fc.function_call, "args", None)
            if args is None and isinstance(fc, dict):
                args = fc.get("args") or (fc.get("function_call") or {}).get("args", {})
            if not isinstance(args, dict):
                args = {}
            return name, args

        model_parts = []
        if text:
            model_parts.append(genai_types.Part.from_text(text=text))
        for fc in function_calls:
            name, args = _fc_name_args(fc)
            model_parts.append(genai_types.Part.from_function_call(name=name, args=args))
        contents.append(genai_types.Content(role="model", parts=model_parts))
        tool_results = []
        for fc in function_calls:
            name, args = _fc_name_args(fc)
            result = _run_tool(name, args)
            tool_results.append((name, result))
            contents.append(genai_types.Content(
                role="tool",
                parts=[genai_types.Part.from_function_response(name=name, response={"result": result})],
            ))
        messages.append({
            "role": "assistant",
            "content": text,
            "tool_calls": [{"id": "", "function": {"name": n, "arguments": json.dumps(a)}} for n, a in (_fc_name_args(fc) for fc in function_calls)],
        })
        for name, result in tool_results:
            messages.append({"role": "tool", "tool_call_id": "", "content": result})
    contents, system_instruction = _openai_messages_to_gemini_contents(messages)
    if stream_placeholder:
        reply = _stream_gemini(client, contents, system_instruction, None, stream_placeholder)
    else:
        config = genai_types.GenerateContentConfig(system_instruction=system_instruction or "")
        final = client.models.generate_content(model=_get_gemini_model(), contents=contents, config=config)
        reply = (final.text or "").strip()
    return (reply, messages)


def call_model_for_agent(
    role_prompt: str,
    speaker: str,
    stream_placeholder=None,
    *,
    build_messages_for_agent: Callable[[str, str, Optional[str]], list],
) -> tuple[str, list]:
    """Call OpenAI or Gemini for the given agent. Returns (reply_text, messages_sent)."""
    if _use_gemini():
        return call_gemini_for_agent(
            role_prompt, speaker, stream_placeholder,
            build_messages_for_agent=build_messages_for_agent,
        )
    return call_openai_for_agent(
        role_prompt, speaker, stream_placeholder,
        build_messages_for_agent=build_messages_for_agent,
    )
