import streamlit as st
import requests
import time

API_URL = "http://localhost:8000/answer"

st.set_page_config(
    page_title="IPC Legal RAG",
    page_icon="⚖️",
    layout="wide",
)

# =========================
# HEADER
# =========================
t0 = time.time()
print("START")
st.markdown(
    """
    <h1 style='text-align:center;'>⚖️ IPC Legal RAG</h1>
    <p style='text-align:center;color:gray;'>
        Hybrid Dense + BM25 + BGE Reranker
    </p>
    """,
    unsafe_allow_html=True,
)

# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# SIDEBAR CONFIG
# =========================

with st.sidebar:

    st.header("⚙️ Retrieval Settings")

    # retriever mode
    retriever_mode = st.selectbox(
        "Retriever Mode",
        ["hybrid_r (prod)", "hybrid_r_eval (debug)"],
        index=0
    )

    # fetch_k (candidate retrieval)
    fetch_k = st.slider(
        "fetch_k (pre-rerank candidates)",
        5, 50, 20
    )

    # final top_k
    top_k = st.slider(
        "top_k (final context to LLM)",
        1, 10, 5
    )

    # min score filter
    min_score = st.slider(
        "min_score (reranker threshold)",
        0.0, 1.0, 0.5, 0.05
    )

    # override eval mode
    if "eval" in retriever_mode:
        min_score = None
        st.caption("Eval mode: min_score disabled")

    st.divider()

    st.markdown("### Default System Values")
    st.code(
        """
fetch_k   = 20
top_k     = 5
min_score = 0.5 (prod)
retriever = hybrid_r
        """
    )
t1 = time.time()
print("retrieval:", t1 - t0)
# =========================
# CHAT HISTORY
# =========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# INPUT
# =========================

query = st.chat_input("Ask IPC legal question...")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Running hybrid RAG pipeline..."):

            try:

                payload = {
                    "query": query,
                    "top_k": top_k,

                    # 🔥 IMPORTANT: pass retrieval config
                    "retriever_config": {
                        "mode": retriever_mode,
                        "fetch_k": fetch_k,
                        "min_score": min_score
                    }
                }
                
                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=None
                )

                if response.status_code != 200:
                    st.error(response.text)
                    st.stop()

                result = response.json()

                answer = result.get("answer", "")
                citations = result.get("citations", [])
                retriever = result.get("retriever", "")
                latency = result.get("latency_ms", 0)

                # streaming display
                placeholder = st.empty()
                text = ""

                for w in answer.split():
                    text += w + " "
                    placeholder.markdown(text)
                    time.sleep(0.005)

                # metrics
                c1, c2 = st.columns(2)

                with c1:
                    st.metric("Retriever", retriever)

                with c2:
                    st.metric("Latency (ms)", f"{latency:.0f}")

                # citations
                if citations:
                    with st.expander("📚 Citations"):
                        for c in citations:
                            st.markdown(
                                f"""
                                **Source:** {c.get('source_path')}  
                                **Section:** {c.get('section_title')}  
                                **Score:** {c.get('score')}
                                """
                            )
                            st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(str(e))