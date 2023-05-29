import streamlit as st
import openai
import os
import tiktoken
from retrying import retry
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
# qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.7), vectordb.as_retriever())

# chat_history = []
# def chat_with_model():
#     question = st.text_input("Ask a question:")
#     # input_text = " "
    
#     # Print chat history
#     # st.subheader("Chat History:")
#     for i, (q, a) in enumerate(chat_history):
#         st.write(f"Q{i+1}: {q}")
#         st.write(f"A{i+1}: {a}")
    
#     result = qa({"question": question, "chat_history": chat_history})
#     chat_history.append((question, result["answer"]))
    
#     st.subheader("Chatbot Response:")
#     st.write(result["answer"])
    

# Streamlit app code
# def main():
#     st.title("Glassnode Chatbot Demo")
#     st.write("Type your question and press Enter:")
#     chat_with_model()
#     # st.experimental_rerun()

# if __name__ == "__main__":
#     main()

# ------ Below is the new version ------ 
system_message = """
    You are glassnodeGPT, a highly sophisticated language model trained to provide onchain advice and insights from the perspective of multiple successful newsletters and tutorials.  
    Your responses should be focused, practical, and direct, mirroring the communication styles of glassnode. Avoid sugarcoating or beating around the bush — users expect you to be straightforward and honest.
    You have access to newsletters and tutorials stored in a vector database. When a user provides a query, you will be provided with snippets of documents that may be relevant to the query. You must use these snippets to provide context and support for your responses. Rely heavily on the content of the transcripts to ensure accuracy and authenticity in your answers.
    Be aware that the chunks of text provided may not always be relevant to the query. Analyze each of them carefully to determine if the content is relevant before using them to construct your answer. Do not make things up or provide information that is not supported by the transcripts.
    Always maintain the signature no-bullshit approach of the knowledge base.
    In your answers, speak confidently as if you were simply speaking from your own knowledge.
"""

vectordb_retriver = vectordb.as_retriever(search_kwargs={"k":3})
# Html format chat bot
bot_msg_container_html_template = '''
<div style='background-color: #FFFFFF; padding: 10px; border-radius: 5px; margin-bottom: 10px; display: flex'>
    <div style="width: 20%; display: flex; justify-content: center">
        <img src="https://cdn.freebiesupply.com/logos/large/2x/harvard-university-logo-svg-vector.svg" style="max-height: 50px; max-width: 50px; border-radius: 50%;">
    </div>
    <div style="width: 80%;">
        $MSG
    </div>
</div>
'''

user_msg_container_html_template = '''
<div style='background-color: #FFFFFF; padding: 10px; border-radius: 5px; margin-bottom: 10px; display: flex'>
    <div style="width: 78%">
        $MSG
    </div>
    <div style="width: 20%; margin-left: auto; display: flex; justify-content: center;">
        <img src="https://yt3.googleusercontent.com/dddKZKUsIKmDM0lrA5sDhgrvYBQKRtJzrJ-NtVM-aAznZpYCJ0wGYIvmvKL3_0uxbYLnRVEA=s176-c-k-c0x00ffffff-no-rj" style="max-width: 50px; max-height: 50px; float: right; border-radius: 50%;">
    </div>    
</div>
'''
def render_chat(**kwargs):
    """
    Handles is_user 
    """
    if kwargs["is_user"]:
        st.write(
            user_msg_container_html_template.replace("$MSG", kwargs["message"]),
            unsafe_allow_html=True)
    else:
        st.write(
            bot_msg_container_html_template.replace("$MSG", kwargs["message"]),
            unsafe_allow_html=True)

    if "figs" in kwargs:
        for f in kwargs["figs"]:
            st.plotly_chart(f, use_container_width=True)

# Define functions to limit the token consumption
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo": 
      num_tokens = 0
      for message in messages:
          num_tokens += 4  
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def ensure_fit_tokens(messages):
    """
    Ensure that total tokens in messages is less than MAX_TOKENS.
    If not, remove oldest messages until it fits.
    """
    total_tokens = num_tokens_from_messages(messages)
    while total_tokens > 4096:
        removed_message = messages.pop(0)
        total_tokens = num_tokens_from_messages(messages)
    return messages

# Initiate session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Construct messages from chat history 
def construct_messages(history):
    messages = [{"role": "system", "content": system_message}]
    
    for entry in history:
        role = "user" if entry["is_user"] else "assistant"
        messages.append({"role":role, "content": entry["message"]})
   # Ensure total tokens do not exceed model's limit
    messages = ensure_fit_tokens(messages)
    return messages 
  
                        
# Function to generate response
def generate_response():
    # Append user's query to history
    st.session_state.history.append({
        "message": st.session_state.prompt,
        "is_user": True
    })
    
    
    # Construct messages from chat history
    messages = construct_messages(st.session_state.history)
    
    # Add the new_message to the list of messages before sending it to the API
    # messages.append(new_message)
    
    # Ensure total tokens do not exceed model's limit
    messages = ensure_fit_tokens(messages)
    
    # Call the Chat Completions API with the messages
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract the assistant's message from the response
    assistant_message = response['choices'][0]['message']['content']
    
    # Append assistant's message to history
    st.session_state.history.append({
        "message": assistant_message,
        "is_user": False
    })
# Take user input
st.title("Glassnode Chatbot Demo")

st.text_input("Enter your prompt:",
              key="prompt",
              placeholder="e.g. 'How can I diversify my portfolio?'",
              on_change=generate_response
              )

# Display chat history
for message in st.session_state.history:
    if message["is_user"]:
        st.write(user_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)
    else:
        st.write(bot_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)
                        

