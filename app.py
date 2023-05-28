import streamlit as st
import openai
import os
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

secrets = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = secrets

# Load the existing persisted database from disk.
persist_path_full = "model/chromadb"
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_path_full, embedding_function=embeddings)

# Define the chatbot function
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.7), vectordb.as_retriever())
chat_history = []
def chat_with_model():
    question = st.text_input("Ask a question:")
    # input_text = " "
    
    # Print chat history
    st.subheader("Chat History:")
    for i, (q, a) in enumerate(chat_history):
        st.write(f"Q{i+1}: {q}")
        st.write(f"A{i+1}: {a}")
    
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    
    st.subheader("Chatbot Response:")
    st.write(result["answer"])
    

# Streamlit app code
def main():
    st.title("Glassnode Chatbot Demo")
    st.write("Type your question and press Enter:")
    chat_with_model()
    # st.experimental_rerun()

if __name__ == "__main__":
    main()
