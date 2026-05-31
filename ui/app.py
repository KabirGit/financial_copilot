"""
Streamlit UI for the Financial Research Copilot.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(
    page_title="Financial Research Copilot",
    page_icon="📊",
    layout="wide",
)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None


def call_research_api(query: str) -> None:
    """POST to the FastAPI backend and update session state."""
    st.session_state.last_error = None
    try:
        response = requests.post(
            f"{API_URL}/research",
            json={"query": query},
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        st.session_state.last_error = "Cannot connect to API. Is the backend running?"
        st.session_state.last_response = None
        return
    except requests.exceptions.Timeout:
        st.session_state.last_error = "Request timed out. The research may still be running."
        st.session_state.last_response = None
        return
    except requests.exceptions.RequestException as exc:
        st.session_state.last_error = f"API request failed: {exc}"
        st.session_state.last_response = None
        return

    if response.status_code == 504:
        detail = response.json().get("detail", {})
        if isinstance(detail, dict):
            st.session_state.last_error = detail.get("error", "Research timed out")
        else:
            st.session_state.last_error = str(detail)
        st.session_state.last_response = None
        return

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
            if isinstance(detail, dict):
                st.session_state.last_error = detail.get("error", str(detail))
            else:
                st.session_state.last_error = str(detail)
        except Exception:
            st.session_state.last_error = response.text or f"HTTP {response.status_code}"
        st.session_state.last_response = None
        return

    data = response.json()
    st.session_state.last_response = data
    if query not in st.session_state.history:
        st.session_state.history.append(query)
    st.session_state.history = st.session_state.history[-5:]


def run_query(query: str) -> None:
    query = query.strip()
    if not query:
        return
    with st.spinner("Analyzing SEC filings..."):
        call_research_api(query)


left, centre, right = st.columns([1, 2, 1])

with left:
    st.header("🔍 Research Query")
    ticker = st.selectbox("Company", ["Any", "AAPL", "MSFT", "GOOGL"])
    question = st.text_area(
        "Your Question",
        placeholder="How did Apple's gross margins change between Q2 and Q3 2024?",
        height=120,
    )
    if st.button("Run Research", type="primary"):
        run_query(question)

    st.divider()
    st.header("📋 History")
    history_slice = list(st.session_state.history[-5:])[::-1]
    for past_query in history_slice:
        label = past_query[:60] + ("…" if len(past_query) > 60 else "")
        if st.button(label, key=f"hist_{past_query}", use_container_width=True):
            run_query(past_query)

    resp = st.session_state.last_response
    if resp:
        confidence = resp.get("confidence", 0.0)
        st.metric("Confidence", f"{confidence:.0%}")
        tools = resp.get("tool_calls_made", [])
        if tools:
            st.caption(f"Tools: {', '.join(tools)}")

with centre:
    st.header("📈 Research Report")
    if st.session_state.last_error:
        st.error(f"Error: {st.session_state.last_error}")
    elif st.session_state.last_response:
        answer = st.session_state.last_response.get("answer", "")
        st.markdown(answer)
        trace = st.session_state.last_response.get("reasoning_trace", "")
        if trace:
            st.caption(f"Reasoning: {trace}")
        confidence = st.session_state.last_response.get("confidence", 1.0)
        if confidence < 0.6:
            st.warning("⚠️ Low confidence — limited data found for this query")
    else:
        st.info("Enter a research question to get started.")

with right:
    st.header("📄 Sources")
    resp = st.session_state.last_response
    sources = (resp or {}).get("sources", [])
    if sources:
        for citation in sources:
            fq = citation.get("fiscal_quarter") or ""
            label = (
                f"{citation.get('ticker', '')} — "
                f"{citation.get('filing_type', '')} "
                f"{citation.get('fiscal_year', '')} {fq}".strip()
            )
            with st.expander(label):
                st.write(f"**Section:** {citation.get('section', '')}")
                st.write(f"**Filed:** {citation.get('date_filed', '')}")
                st.write(f"**Chunk ID:** {citation.get('chunk_id', '')}")
    else:
        st.caption("No sources to display.")
