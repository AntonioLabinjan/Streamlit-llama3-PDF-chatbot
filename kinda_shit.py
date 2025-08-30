import io
import json
import time
from typing import List, Tuple

import streamlit as st

# --- PDF extraction helpers -------------------------------------------------

def extract_pdf_text(file_bytes: bytes) -> Tuple[str, List[str]]:
    """Return full text and per-page texts.
    Tries pypdf first (modern), then PyPDF2 as a fallback.
    """
    pages = []
    full_text = ""

    # Try pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(file_bytes))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(text)
        full_text = "\n".join(pages)
        return full_text, pages
    except Exception:
        pass

    # Fallback to PyPDF2
    try:
        import PyPDF2  # type: ignore
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        full_text = "\n".join(pages)
        return full_text, pages
    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF text: {e}")


# --- LangChain / RAG setup --------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=model_name)


def build_vectorstore(file_bytes: bytes, filename: str, embed_model: str, chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    full_text, pages = extract_pdf_text(file_bytes)

    # Keep the original page numbers in metadata for citations
    docs: List[Document] = []
    for idx, page_text in enumerate(pages):
        docs.append(Document(page_content=page_text, metadata={"source": filename, "page": idx + 1}))

    # Split across pages for better recall
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)

    embeddings = get_embeddings(embed_model)
    vs = FAISS.from_documents(split_docs, embeddings)

    meta = {
        "filename": filename,
        "num_pages": len(pages),
        "num_chunks": len(split_docs),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    return vs, meta


def retrieve_context(vs, query: str, k: int):
    # Return (formatted_context, sources_list)
    results = vs.similarity_search(query, k=k)
    sources = []
    context_blocks = []
    for i, d in enumerate(results, start=1):
        src = f"{d.metadata.get('source','file')} p.{d.metadata.get('page','?')}"
        sources.append(src)
        block = f"[Source {i}: {src}]\n{d.page_content.strip()}"
        context_blocks.append(block)
    return "\n\n".join(context_blocks), sources


def stream_generate(model_name: str, system_prompt: str, question: str, history: List[dict], context: str, temperature: float):
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_ollama import OllamaLLM

    # FIX: include context explicitly in the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("system", "CONTEXT (from retrieved PDF chunks):\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])

    llm = OllamaLLM(model=model_name, temperature=temperature)

    chain = (
        {
            "history": lambda x: history,
            "question": lambda x: question,
            "context": lambda x: context,
            "system_prompt": lambda x: system_prompt,
        }
        | prompt
        | llm
    )

    for chunk in chain.stream({}):
        yield chunk


# --- UI --------------------------------------------------------------------

st.set_page_config(page_title="AI PDF Chat (Fast RAG)", page_icon="📄🤖", layout="wide")

st.title("AI ChatBot with PDF Support 📄🤖 — Fast RAG Edition")

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.text_input("Ollama model", value="llama3")
    embed_model = st.text_input("Embedding model", value="nomic-embed-text")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
    k = st.slider("Top-K Chunks", 1, 10, 4)
    chunk_size = st.number_input("Chunk size", 256, 4000, 1200, step=64)
    chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 150, step=10)

    st.markdown("---")
    st.caption("Upload one or more PDFs. They'll be combined into a single searchable knowledge base for this session.")
    files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("kb_meta", None)
        st.rerun()

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "kb_meta" not in st.session_state:
    st.session_state.kb_meta = []

# Build / extend the vector store when files are uploaded
if files:
    from langchain_community.vectorstores import FAISS

    for f in files:
        file_bytes = f.read()
        with st.spinner(f"Indexing {f.name} …"):
            vs, meta = build_vectorstore(file_bytes, f.name, embed_model, int(chunk_size), int(chunk_overlap))

        # Merge into one FAISS index across multiple PDFs
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = vs
        else:
            st.session_state.vectorstore.merge_from(vs)
        st.session_state.kb_meta.append(meta)

    st.success("Knowledge base ready! Ask questions below.")

# Show KB status
with st.expander("📚 Knowledge base status", expanded=False):
    if st.session_state.vectorstore is None:
        st.info("No PDFs indexed yet.")
    else:
        total_chunks = sum(m["num_chunks"] for m in st.session_state.kb_meta)
        total_pages = sum(m["num_pages"] for m in st.session_state.kb_meta)
        st.write({
            "files": [m["filename"] for m in st.session_state.kb_meta],
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        })

# Display prior messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"]) 

# Chat input
user_q = st.chat_input("Ask a question about your PDFs …")
if user_q:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_q})

    # Retrieve context
    retrieved = ""
    sources = []
    if st.session_state.vectorstore is not None:
        retrieved, sources = retrieve_context(st.session_state.vectorstore, user_q, int(k))
    else:
        retrieved = "(No documents indexed yet.)"

    # System prompt with instructions & citation expectations
    system_prompt = (
        "You are a helpful analyst. Answer ONLY using the provided CONTEXT when possible.\n"
        "If the answer isn't clearly present, say so. Be concise. If relevant, include a short bullet list of key points."
    )

    # Assistant streaming output
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        start = time.time()
        for token in stream_generate(model_name, system_prompt, user_q, st.session_state.messages, retrieved, temperature):
            full += token
            placeholder.markdown(full)
        dur = time.time() - start

        # Append a tiny sources/citations footer
        if sources:
            full += "\n\n---\n**Sources**: " + ", ".join(sorted(set(sources)))
            placeholder.markdown(full)
            st.caption(f"Answered in {dur:.1f}s • Using top-{k} chunks")

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": full})

# Download conversation
if st.session_state.get("messages"):
    transcript = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
    st.download_button(
        label="💾 Download chat JSON",
        data=transcript,
        file_name="chat_transcript.json",
        mime="application/json",
    )

# Footer
st.markdown(
    "<div style='text-align:center; opacity:0.6; font-size:12px'>"
    "Built with Streamlit + LangChain + Ollama • RAG over PDFs • Supports multiple files & citations"
    "</div>", unsafe_allow_html=True
)
