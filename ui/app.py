"""Streamlit chat UI. Talks to the FastAPI backend over HTTP only:
no models, no index and no API keys in this container.

Run locally:  uv run --with streamlit streamlit run ui/app.py   (API on :8000)
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES = [
    "How many casual leaves do I get?",
    "Can I claim my home internet bill?",
    "What is my notice period if I'm on probation?",
    "If I resign, is my unused earned leave paid out?",
    "Is there a relocation allowance?",
]

st.set_page_config(page_title="HR policy assistant", page_icon="📘", layout="wide")


def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=90, **kwargs)
    except requests.ConnectionError:
        return None, f"Can't reach the backend at {API_URL}. Start it with: uv run uvicorn hr_assistant.api:app"
    except requests.Timeout:
        return None, "The backend took too long to respond. Please try again."
    if r.ok:
        return r.json(), None
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    if isinstance(detail, list):
        detail = "; ".join(d.get("msg", str(d)) for d in detail)
    return None, str(detail)


def render_answer(body: dict) -> None:
    status = body["status"]
    if status == "answered":
        st.markdown(body["answer"])
        if body.get("disclaimer"):
            st.info(body["disclaimer"])
        if body["cited_sources"]:
            with st.expander(f"Sources ({', '.join(body['citations'])})"):
                for s in body["cited_sources"]:
                    st.markdown(f"**{s['citation']}** · {s['heading']}")
                    st.caption(s["text"])
    elif status == "not_covered":
        st.warning(body["answer"])
        if body["sources"]:
            with st.expander("Closest sections found (they don't answer the question)"):
                for s in body["sources"]:
                    st.markdown(f"**{s['citation']}** · {s['heading']}")
    else:  # declined
        st.markdown(body["answer"])


def page_chat() -> None:
    st.header("Ask about HR policies")
    st.caption("Answers come only from Kestrel's HR policies, with the sections they're based on.")

    history = st.session_state.setdefault("history", [])
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            if turn.get("error"):
                st.error(turn["error"])
            else:
                render_answer(turn["body"])

    pending = st.session_state.pop("example", None)
    question = st.chat_input("Ask a question, e.g. How much notice do I need to give?") or pending
    if not question:
        if not history:
            st.write("Try one of these:")
            for ex in EXAMPLES:
                if st.button(ex, key=f"ex_{ex}"):
                    st.session_state["example"] = ex
                    st.rerun()
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the policies..."):
            body, error = api("POST", "/ask", json={"question": question})
        if error:
            st.error(error)
        else:
            render_answer(body)
    history.append({"question": question, "body": body, "error": error})


def page_policies() -> None:
    st.header("Browse policies")
    sections, error = api("GET", "/policies")
    if error:
        st.error(error)
        return
    df = pd.DataFrame(sections)
    policy = st.selectbox("Policy", sorted(df["policy"].unique()))
    for _, row in df[df["policy"] == policy].iterrows():
        st.markdown(f"**{row['citation']}** · {row['heading'].split(' > ', 1)[-1]}")
        st.write(row["text"])


def page_dashboard() -> None:
    st.header("Usage, cost and latency")
    data, error = api("GET", "/telemetry")
    if error:
        st.error(error)
        return
    t = data["summary"]
    if not t or not t["requests"]:
        st.write("No questions yet. Ask something in the chat to see data here.")
        return
    cost = t["avg_cost_per_request_usd"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions", t["requests"])
    c2.metric("Average latency", f"{t['avg_latency_ms'] / 1000:.1f} s")
    c3.metric("p95 latency", f"{t['p95_latency_ms'] / 1000:.1f} s")
    c4.metric("Cost per question", "n/a" if cost is None else f"${cost:.5f}")

    st.subheader("Outcomes")
    st.bar_chart(pd.Series(t["by_status"], name="questions"))

    if t["steps"]:
        st.subheader("LLM steps")
        steps = pd.DataFrame(t["steps"])
        st.bar_chart(steps.set_index("step")[["avg_latency_ms", "p95_latency_ms"]])
        st.dataframe(steps, hide_index=True, width="stretch")

    st.subheader("Recent questions")
    st.dataframe(pd.DataFrame(data["recent"])[["created_at", "question", "status", "category", "latency_ms"]],
                 hide_index=True, width="stretch")


PAGES = {"Ask a question": page_chat, "Browse policies": page_policies, "Usage and cost": page_dashboard}

with st.sidebar:
    st.title("HR policy assistant")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    if page == "Ask a question" and st.session_state.get("history"):
        if st.button("Clear conversation"):
            st.session_state["history"] = []
            st.rerun()
    st.caption(f"Backend: {API_URL}")

PAGES[page]()
