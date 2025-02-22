import streamlit as st
import PyPDF2
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Define the prompt template
template = """ 
Answer the question below based on the provided conversation history and document contents.

Here is the conversation history: {context}

Question: {question}

Answer:  
"""

# Initialize the model
model = OllamaLLM(model='llama3')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit UI
st.title("AI ChatBot with PDF Support 📄🤖")
st.write("Upload a PDF and ask questions about it!")

# Session state to maintain chat history and document context
if "context" not in st.session_state:
    st.session_state.context = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# File upload for AI analysis
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        pdf_text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

        # Store extracted text in session state (limit to 2000 chars to prevent overload)
        st.session_state.pdf_text = pdf_text[:2000]  # Adjust if needed
        st.session_state.context += f"\nExtracted PDF content: {st.session_state.pdf_text}\n"

        st.success("PDF uploaded successfully! You can now ask questions about it.")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask a question about the document...")
if user_input:
    with st.spinner("Thinking..."):
        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Stream AI response
        response_stream = chain.stream({"context": st.session_state.context, "question": user_input})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for chunk in response_stream:
                full_response += chunk
                message_placeholder.markdown(full_response)  # Stream the response live

        # Append AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Update context
        st.session_state.context += f"\nUser: {user_input}\nAI: {full_response}\n"
