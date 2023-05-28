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



# Create embeddings using OpenAI API
# embeddings = OpenAIEmbeddings()

# print("Loading Existing DB!")
# embeddings_dir_name = "model"
# db_filename = "chromadb"

# if not os.path.exists(embeddings_dir_name):
#     os.makedirs(embeddings_dir_name)

persist_path_full = "model/chromadb"

# Load the existing persisted database from disk.
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_path_full, embedding_function=embeddings)
# load_vectordb = Annoy.load(persist_path_full, embeddings = embeddings_function)

# Define the chatbot function
def chat_with_model(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
    )
    return response.choices[0].message.content

def find_related_documents(query):
    docs_with_scores_and_content_texts = vectordb.similarity_search_with_score(query)[:3]
    return [doc.page_content for doc,_ in docs_with_scores_and_content_texts]


# Streamlit app code
def main():
    st.title("Glassnode Chatbot Demo")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

        initial_message ={
            "role": "system",
            "content": "You are a smart information retrieval AI. Below you will see questions from user, and several related items for the question. Based on the documents, answer the question. If you cannot answer the question based on the documents, just say you don't know. Don't try to make up answers."
            }

        st.session_state['messages'].append(initial_message)

    chat_history_container = st.container()

    with chat_history_container:
        for message in st.session_state["messages"]:
            if(message["role"]=="user"):
                st.markdown(f"**You**: {message['content']}")
            elif (message["role"]=="assistant"):
                st.markdown(f"_Assistant_: {message['content']}")

        st.write("\nType your message below:")
        user_input = st.text_input("Enter", key=len(st.session_state["messages"]))

    if user_input:
        # Reset the chat history
        st.session_state['messages'] = []
        # Append user message to the conversation
        new_message = {"role": "user", "content": user_input}
        st.session_state['messages'].append(new_message)

        # Find related documents based on user input and add them as messages
        related_documents = find_related_documents(user_input)

        if len(related_documents) > 0:
            doc_messages = [{"role": "system", "content": doc_text} for doc_text in related_documents]
            st.session_state["messages"].extend(doc_messages)

        # Get chatbot's response and append it to the conversation
        bot_response = chat_with_model(st.session_state['messages'])

        assistant_response = {"role": "assistant", "content": bot_response}
        st.session_state["messages"].append(assistant_response)

        # We need to manually rerun Streamlit script so that Streamlit can process new state & display it
        st.experimental_rerun()

if __name__ == "__main__":
    main()
