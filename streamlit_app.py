import streamlit as st
from src.text_extractor import ExtractText
from src.vector import Embedder
from langchain_groq import ChatGroq
import tempfile
from langchain_groq import ChatGroq
import time
import os
from dotenv import load_dotenv

load_dotenv()
st.title("📚 PDF Q&A Assistant")
st.write("Upload a PDF and ask questions about it in a conversational way.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []
if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = "ALL PDFs"
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

with st.sidebar:
    st.header("Instructions")
    st.write("1. Upload a PDF")
    st.write("2. Wait for processing")
    st.write("3. Ask questions!")
    st.divider()
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.session_state.uploaded_pdfs:
        st.write("Uploaded PDFs:")
        for pdf in st.session_state.uploaded_pdfs:
            st.write(f"📄 {pdf}")
        
        selected_pdf = st.selectbox(
            "Which PDF do you want to query?",
            ["ALL PDFs"] + st.session_state.uploaded_pdfs
        )
        st.session_state.selected_pdf = selected_pdf

    if st.session_state.messages:
        if st.sidebar.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()


if uploaded_file and uploaded_file.name not in st.session_state.uploaded_pdfs:
    # Save uploaded file temporarily to disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing PDF..."):
        extractor = ExtractText(tmp_path, filename=uploaded_file.name)
        extractor.extract_text()
        if len(extractor.chunks) == 0:
            st.error("Could not extract text from this PDF. It may be a scanned image.")
            st.stop()
        try:
            st.session_state.embedder = Embedder()
        except Exception as e:
            st.error("Could not connect to the database. Please try again.")
            st.stop()
        if not st.session_state.embedder.is_embedded(uploaded_file.name):
            start = time.time()
            st.session_state.embedder.embedder(chunks=extractor.chunks, metadatas=extractor.metadatas)
            end = time.time()
            print(f"Chunks: {len(extractor.chunks)}, Embedding time: {end - start:.2f} seconds")
    st.session_state.uploaded_pdfs.append(uploaded_file.name)
    st.session_state.all_chunks.extend(extractor.chunks)
    st.success("PDF processed! Ask a question.")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if len(st.session_state.uploaded_pdfs) > 0:
    st.info(f"Querying: {st.session_state.selected_pdf}")
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        start = time.time()
        results = st.session_state.embedder.get_answer(
                                    prompt,
                                    selected_pdf=st.session_state.selected_pdf,
                                    session_pdfs=st.session_state.uploaded_pdfs
                                )
        
        bm25_chunks = st.session_state.embedder.bm25_search(
        st.session_state.all_chunks, 
        prompt
    )
        combined_chunks = results["documents"][0] + bm25_chunks
        unique_chunks = list(dict.fromkeys(combined_chunks))
        unique_chunks = st.session_state.embedder.re_ranking(unique_chunks, prompt)

        if not unique_chunks:
            with st.chat_message("assistant"):
                st.write("I couldn't find relevant information in the document to answer this question.")
        else:
            context = "\n\n---\n\n".join(unique_chunks)
            pages = [f"{m['Source']} - Page {m['Pages']}" for m in results["metadatas"][0]]
        
            llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.5)
            history = ""
            for message in st.session_state.messages[-6:]:  
                role = "User" if message["role"] == "user" else "Assistant"
                history += f"{role}: {message['content']}\n"
            try:
                answer = llm.stream(f"""You are a helpful tutor assisting a student with their textbook.
                                        Use the context below as your PRIMARY source of knowledge.
                                        Do NOT add outside knowledge or information not present in the context.
                                        Explain concepts conversationally — like explaining to a friend.
                                        Use conversation history to understand follow up questions.
                                        When writing code, base it on examples from the context but clearly mention it may need testing.
                                        Do not make up specific numbers, statistics or package names that aren't in the context.
                                        If the answer is truly not in the context, say "I don't know"
                                    You are a knowledgeable technical assistant. Your goal is to provide clear, 
                                    accurate answers based on the provided CONTEXT and CONVERSATION HISTORY.

                                    GUIDELINES:
                                    1. USE THE CONTEXT: Prioritize information found in the document.
                                    2. INFERENCE: You are allowed to use logical reasoning to connect ideas within the context to provide a complete answer.
                                    3. BE CONVERSATIONAL: Explain concepts simply, like you are talking to a colleague.
                                    5. IF MISSING: If the context doesn't have the specific answer, don't just say "I don't know." Instead, tell the user what related information is available in the document.".

                                    CONTEXT: {context}

                                    CONVERSATION HISTORY: {history}

                                    QUESTION: {prompt}

                                    ANSWER:""")
                end = time.time()
                print(f"Response time: {end - start:.2f} seconds")
                #response = f"{answer.content}\n\n*Source: Pages {pages}*"
                with st.chat_message("assistant"):
                    response_text = st.write_stream(answer)
                    st.write(f"*Source: Pages {pages}*")
                st.session_state.messages.append({"role": "assistant", "content": f"{response_text}\n\n*Source: Pages {pages}*"})
            except Exception as e:
                print(f"Error: {e}")
                with st.chat_message("assistant"):
                    st.write("Unable to get a response right now. Please try again.")
                st.session_state.messages.append({"role": "assistant", "content": "Unable to get a response right now. Please try again."})