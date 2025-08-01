import os
import base64
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# ✅ Use secrets.toml for API key
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["apikey"]

# Configure Streamlit page
st.set_page_config(
    page_title="MOVIT PRODUCTS LIMITED HR Assistant",
    page_icon="📘",
    layout="wide"
)

# Convert image to base64
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# Set background image
def set_exact_background(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    css = f"""
    <style>
    html, body, .stApp {{
        background: url("data:image/png;base64,{bin_str}") no-repeat left top fixed;
        background-size: 100% 100%;
        height: 100vh;
        width: 100vw;
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

# Set your image as the background
set_exact_background("image.png")  # Make sure this image is uploaded to your repo

# Begin main content overlay
st.markdown('<div class="content-overlay">', unsafe_allow_html=True)
st.markdown("### 📘 MOVIT PRODUCTS LIMITED HR Assistant", unsafe_allow_html=True)
st.markdown("_Answers are based only on HR Manual and the Staff Rotation & Transfer Policy._")

# Load FAISS vectorstore
@st.cache_resource
def load_vectorstore():
    persist_path = "faiss_index_combined"
    embeddings = OpenAIEmbeddings()

    if os.path.exists(persist_path):
        return FAISS.load_local(
            persist_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    with st.spinner("Processing HR documents (first time only)..."):
        hr_loader = PyPDFLoader("HR-Manual.pdf")
        hr_docs = hr_loader.load()

        staff_loader = PyPDFLoader("Staff_Rotation_Transfer_Policy.pdf")
        staff_docs = staff_loader.load()

        all_docs = hr_docs + staff_docs
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(persist_path)
        return vectorstore

# Load vectorstore and model
vectorstore = load_vectorstore()
llm = OpenAI(temperature=0.3, max_tokens=1024)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Input box
query = st.text_input("🔎 Ask something from the HR Manual or Staff Rotation & Transfer Policy:")

# Handle input and output
if query:
    prompt = (
        f"You are a helpful HR assistant. Use only the HR Manual and the Staff Rotation & Transfer Policy to answer. "
        f"Give a detailed and structured response using headings like Definitions, Policies, Procedures, and Examples where applicable.\n\n"
        f"Question: {query}"
    )

    with st.spinner("Analyzing HR documents..."):
        result = qa_chain.run(prompt)
        st.markdown('<div class="response-box">', unsafe_allow_html=True)
        st.markdown("### ✅ Answer:")
        st.write(result)
        st.markdown('</div>', unsafe_allow_html=True)

# Close content overlay
st.markdown('</div>', unsafe_allow_html=True)
