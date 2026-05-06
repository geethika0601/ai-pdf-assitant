
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main app background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #161B22;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px;
    border: 1px solid #30363D;
}

/* Chat content */
[data-testid="stChatMessageContent"] {
    color: white;
    font-size: 16px;
}

/* Input box */
.stChatInputContainer {
    border-top: 1px solid #30363D;
    background-color: #0E1117;
}

/* Buttons */
.stButton button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 16px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #161B22;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #30363D;
}

/* Headings */
h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- GROQ CLIENT ----------------

client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"]
    base_url="https://api.groq.com/openai/v1"
)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("📄 AI PDF Assistant")

    st.markdown("""
    Upload multiple PDFs and ask intelligent
    questions across documents.
    """)

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------- CHAT MEMORY ----------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- LOAD EMBEDDING MODEL ----------------

@st.cache_resource

def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ---------------- PROCESS PDFs ----------------

chunks = []
chunk_sources = []

if uploaded_files:

    with st.spinner("📚 Processing PDFs..."):

        for uploaded_file in uploaded_files:

            pdf_reader = PdfReader(uploaded_file)

            text = ""

            for page_num, page in enumerate(pdf_reader.pages):

                extracted = page.extract_text()

                if extracted:
                    text += extracted

            # Chunking
            chunk_size = 800
            overlap = 100

            start = 0

            while start < len(text):

                end = start + chunk_size

                chunk = text[start:end]

                chunks.append(chunk)

                chunk_sources.append({
                    "file": uploaded_file.name
                })

                start += chunk_size - overlap

    st.success(f"✅ Processed {len(uploaded_files)} PDF(s)")

    st.info(f"📚 Documents Loaded: {len(uploaded_files)} | 🧩 Chunks Created: {len(chunks)}")

# ---------------- CREATE VECTOR DATABASE ----------------

if chunks:

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

# ---------------- MAIN PAGE ----------------

st.markdown("""
# 🤖 AI Research Assistant

### Chat with multiple PDFs using AI
""")

# ---------------- DISPLAY CHAT HISTORY ----------------

for role, message in st.session_state.chat_history:

    with st.chat_message(role):
        st.markdown(message)

# ---------------- USER INPUT ----------------

question = st.chat_input(
    "Ask a question about your PDFs..."
)

if question and chunks:

    # Save user question
    st.session_state.chat_history.append(
        ("user", question)
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("🧠 Analyzing documents..."):

            # Embed question
            question_embedding = embedding_model.encode([question])

            # Search top chunks
            D, I = index.search(
                np.array(question_embedding),
                k=5
            )

            relevant_chunks = []
            sources = []

            for idx in I[0]:
                relevant_chunks.append(chunks[idx])
                sources.append(chunk_sources[idx]["file"])

            context = "\n".join(relevant_chunks)

            # Previous memory
            memory = "\n".join([
                f"{role}: {msg}"
                for role, msg in st.session_state.chat_history
            ])

            # Prompt
            prompt = f"""
            You are an intelligent AI research assistant.

            Answer ONLY from the provided context.

            If the answer is not found,
            say:
            'I could not find that information in the uploaded documents.'

            Keep answers clear and concise.

            Previous conversation:
            {memory}

            Context:
            {context}

            User question:
            {question}
            """

            # Call Groq API
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content

            st.markdown(answer)

            # Sources
            unique_sources = list(set(sources))

            st.markdown("---")
            st.markdown("### 📚 Sources")

            for src in unique_sources:
                st.markdown(f"- {src}")

            # Save assistant answer
            st.session_state.chat_history.append(
                ("assistant", answer)
            )

elif question and not uploaded_files:

    st.warning("⚠️ Please upload at least one PDF.")

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit, FAISS, Sentence Transformers, and Groq"
)

