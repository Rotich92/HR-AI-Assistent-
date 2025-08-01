import os
import base64
import streamlit as st
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# ✅ Load API key securely from .streamlit/secrets.toml
API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = API_KEY

# ✅ Streamlit settings
st.set_page_config(page_title="MOVIT PRODUCTS LIMITED HR Assistant", page_icon="📘", layout="wide")

# ✅ Background image
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def set_exact_background(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    css = f"""
    <style>
    html, body, .stApp {{
        background: url("data:image/png;base64,{bin_str}") no-repeat left top fixed;
        background-size: 100% 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
        font-family: "Segoe UI", sans-serif;
    }}
    .content-overlay {{
        position: absolute;
        top: 35%;
        left: calc(50% + 1in);
        transform: translate(-50%, -35%);
        width: 60%;
        background: rgba(255, 255, 255, 0);
    }}
    input[type="text"] {{
        font-size: 16px !important;
        text-align: center;
    }}
    .response-box {{
        margin-top: 1.5rem;
        margin-left: 2in;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 1rem;
        border-radius: 10px;
        width: 80%;
        text-align: left;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ✅ Set background
set_exact_background("image.png")

# ✅ App header
st.markdown('<div class="content-overlay">', unsafe_allow_html=True)
st.markdown("### 📘 MOVIT PRODUCTS LIMITED HR Assistant", unsafe_allow_html=True)
st.markdown("_Answers are based only on HR Manual and the Staff Rotation & Transfer Policy._")

# ✅ Load or create FAISS vectorstore
@st.cache_resource
def load_vectorstore():
    persist_path = "faiss_index_combined"
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

    if os.path.exists(persist_path):
        return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)

    with st.spinner("🔄 Processing HR documents (first time only)..."):
        hr_docs = PyPDFLoader("HR-Manual.pdf").load()
        staff_docs = PyPDFLoader("Staff_Rotation_Transfer_Policy.pdf").load()
        all_docs = hr_docs + staff_docs

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)

        try:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(persist_path)
            return vectorstore
        except Exception as e:
            st.error(f"🚫 Embedding error: {str(e)}")
            st.stop()

# ✅ Load everything
vectorstore = load_vectorstore()
llm = OpenAI(temperature=0.3, max_tokens=1024, openai_api_key=API_KEY)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ✅ Input box
query = st.text_input("🔎 Ask something from the HR Manual or Staff Rotation & Transfer Policy:")

# ✅ Improved structured prompt template
def build_prompt(user_query):
    return f"""
You are a highly accurate and professional HR assistant for Movit Products Limited.

Use only the content from the HR Manual and the Staff Rotation & Transfer Policy to answer the question below.

🧠 Follow these strict rules:
- Begin with a short and clear **Answer (Summary)** that reflects the exact policy if applicable.
- If the question involves specific terms (e.g., working hours, leave days, weekends, benefits), **quote the relevant section of the policy verbatim** in the next section.
- Do NOT generalise, assume, or invent information. Stick exactly to the provided documents.

📄 Use this structure:
**Answer (Summary):** Short and clear direct answer in natural language.
**Details from Policy:** Verbatim or clearly referenced excerpt from the official documents.
**Clarification/Examples:** MUST be included. Use this section to explain the meaning, intent, or practical application of the policy language in simple terms. Clarify ambiguous phrasing or potential confusion for employees.

You MUST include a detailed and structured breakdown using all of the following headings:
- **Definitions**: Define any relevant terms used in the policy.
- **Policies**: State the governing policies as documented.
- **Procedures**: Outline any step-by-step actions if applicable.
- **Examples**: Give examples or interpretations if helpful.

Within each section (Definitions, Policies, Procedures, and Examples), list all relevant items and points from the documents in full detail. Ensure no important content is omitted. Do not summarise or generalise any part of the policy language. Every sub-point, clause, or instruction under a given section should be explicitly stated.

Question: {user_query}
"""

# ✅ Generate response
if query:
    with st.spinner("🤖 Analyzing HR documents..."):
        try:
            final_prompt = build_prompt(query)
            result = qa_chain.run(final_prompt)
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            st.markdown("### ✅ Answer:")
            st.markdown(result, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"🚫 LLM processing failed: {str(e)}")

# ✅ Close content overlay
st.markdown('</div>', unsafe_allow_html=True)
