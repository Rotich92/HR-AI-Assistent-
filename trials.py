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
- Search for and include **all document sections that relate to the query**.
- Give **priority** to exact bullet points, numbered lists, and structured clauses from the documents. These must be shown **exactly as written**.
- Do not summarise, rephrase, or omit any item within a list or numbered section.
- Do NOT generalise, assume, or invent information. You must work strictly within the document content.

📄 Use this response structure:
**Answer (Summary):** A short but direct answer in natural language, based only on quoted policy.
**Relevant Policy Section(s):** Copy the **full content** of all identified relevant sections, including **all numbered items, bullet points, tables, or formatting** used in the source.
**Clarification/Examples:** Explain the content in simple terms or how it applies in practice. You may reference examples only if consistent with the quoted content.

You MUST break down the response using these headings:
- **Definitions**: Define any technical or policy terms from the source section(s).
- **Policies**: Present all full policy clauses exactly as written.
- **Procedures**: If any processes or approval steps are included, list all of them in order.
- **Examples**: Share exact or clearly derived use cases if mentioned. Do not invent examples.

🛑 Do not ignore or summarise any bullet points or numbered procedures — they must be listed as-is.

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
