from .config import genai, SYSTEM_PROMPT
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_rag_response(query, tone="informative", context=None):
    # If context is provided, create a temporary vector store
    if context:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(context)
        vectorstore = FAISS.from_texts(chunks, embeddings)
    else:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = vectorstore.similarity_search(query, k=1)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"{SYSTEM_PROMPT}\n\nTone: {tone}\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

__all__ = ["get_rag_response"]