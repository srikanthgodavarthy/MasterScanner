"""
pages/agent.py — Agent tab.

A chat interface, backed by OpenAI function calling, that can answer
questions about Nifty 500 stocks using:
  • this session's live scan results (Leadership/Conviction/Entry Quality,
    signal class, entry/SL/targets)
  • persisted Trinity data in Supabase (lifecycle stage, watchlist,
    setup plans, backtest stats)
  • general market data via yfinance (fundamentals, news, price performance)
  • the model's own general market/finance knowledge for everything else

Tool definitions and implementations live in utils/agent_tools.py; this file
is just the chat UI + the tool-calling loop.
"""

from __future__ import annotations

import json

import streamlit as st

from utils.openai_client import get_client, get_model, _is_available
from utils.agent_tools import TOOLS, call_tool

MAX_TOOL_ITERS = 5

SYSTEM_PROMPT = """You are the Trinity Agent, embedded in a Nifty 500 swing-trading \
scanner app. You help the user understand stocks in the Nifty 500 universe by combining:

1. Live scan data (this browser session only) — signal class, Leadership/Conviction/Entry \
Quality/Extension scores, entry/SL/target levels. Only available if the user has run a scan \
on the Live Scanner tab in this session.
2. Persisted Trinity data (Supabase) — lifecycle stage history, watchlist, locked setup \
plans, backtest win-rate/R stats. Available across sessions.
3. General market data (yfinance) — fundamentals, recent news, price performance.
4. Your own general knowledge of markets, sectors, and macro context for anything not covered \
by the above.

Rules:
- Always prefer calling a tool over guessing when the question is about a specific stock's \
score, stage, plan, or fundamentals — don't invent numbers.
- Be upfront and brief when a tool says data isn't available (e.g. no live scan run yet, \
Supabase not configured) rather than papering over it.
- Be clear about which parts of an answer come from Trinity's own scoring vs. general \
market commentary from your own knowledge.
- This is not investment advice — you can discuss scores, levels, and context, but frame \
things as information, not recommendations to buy/sell.
- Keep answers tight and scannable: short paragraphs, bullets for multiple stocks, no filler.
"""


def _render_history(messages: list[dict]) -> None:
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        with st.chat_message("user" if role == "user" else "assistant",
                              avatar="🧑" if role == "user" else "⚡"):
            st.markdown(content)


def _run_turn(client, model: str, messages: list[dict]) -> None:
    """Runs the tool-calling loop for one user turn, mutating `messages` in place."""
    for _ in range(MAX_TOOL_ITERS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
            )
        except Exception as exc:
            messages.append({"role": "assistant",
                              "content": f"⚠️ OpenAI request failed: {exc}"})
            return

        choice = resp.choices[0].message
        msg_dict = choice.model_dump(exclude_none=True)
        messages.append(msg_dict)

        tool_calls = msg_dict.get("tool_calls")
        if not tool_calls:
            return  # final answer, no more tools requested

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            with st.spinner(f"🔧 {fn_name}({', '.join(f'{k}={v}' for k, v in args.items())})"):
                result = call_tool(fn_name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, default=str),
            })

    messages.append({"role": "assistant",
                      "content": "⚠️ Hit the tool-call limit for this turn — try rephrasing "
                                 "or asking about one stock at a time."})


def render(settings: dict | None = None) -> None:
    st.markdown("""
    <div style="margin-bottom:0.6rem;">
        <p style="font-size:1.4rem;font-weight:700;color:#e6edf3;margin:0;">🤖 Agent</p>
        <p style="font-size:0.85rem;color:#8b949e;margin:2px 0 0 0;">
            Ask about any Nifty 500 stock — live scores, lifecycle, fundamentals, news
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not _is_available():
        st.markdown(
            '<span class="status-pill" style="background:rgba(248,81,73,0.1);'
            'border:1px solid rgba(248,81,73,0.35);color:#f85149">'
            '● OpenAI not configured</span>',
            unsafe_allow_html=True,
        )
        st.code('OPENAI_API_KEY = "sk-..."\n# optional, defaults to gpt-4o-mini\nOPENAI_MODEL = "gpt-4o-mini"',
                language="toml")
        st.caption("Add the above to `.streamlit/secrets.toml`, then reload.")
        return

    client = get_client()
    model = get_model()

    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

    top_c1, top_c2 = st.columns([6, 1])
    with top_c1:
        st.caption(f"Model: `{model}` · Data: live scan (this session) + Supabase + yfinance")
    with top_c2:
        if st.button("🗑️ Clear", width='stretch'):
            st.session_state["agent_messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.rerun()

    messages = st.session_state["agent_messages"]
    _render_history(messages)

    if len(messages) == 1:
        st.markdown(
            '<div style="color:#64748b;font-size:0.85rem;padding:0.5rem 0;">'
            'Try: "What\'s RELIANCE\'s CV1 score right now?" · "Show me today\'s Elite Opportunity '
            'signals" · "What lifecycle stage is TCS in?" · "Any news on INFY?" · "What sector is '
            'AARTIIND in and what\'s its PE?"</div>',
            unsafe_allow_html=True,
        )

    prompt = st.chat_input("Ask about a Nifty 500 stock…")
    if prompt:
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Thinking…"):
                _run_turn(client, model, messages)
            final = messages[-1]
            if final.get("role") == "assistant" and final.get("content"):
                st.markdown(final["content"])
        st.session_state["agent_messages"] = messages
