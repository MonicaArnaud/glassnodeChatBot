import pandas as pd
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import chromadb
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = "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"
# Solve VPN problem
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

# Read Excel data
# df = pd.read_excel('product_list.xlsx', engine='openpyxl')
glassnode_loader = DirectoryLoader('./docs/', glob="*.pdf")

glassnode_docs = glassnode_loader.load()


# Display first 5 rows of data
# print(df.head())

# from langchain.document_loaders import DataFrameLoader

# loader = DataFrameLoader(df, page_content_column="Product Details")
# documents = loader.load()

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Create embeddings using OpenAI API
embeddings = OpenAIEmbeddings()

embeddings_dir_name = "model"
db_filename = "chromadb"

if not os.path.exists(embeddings_dir_name):
    os.makedirs(embeddings_dir_name)

persist_path_full=os.path.join(embeddings_dir_name, db_filename)

if not os.path.exists(persist_path_full):
    print("Creating new DB!")
    # Create a Chroma instance with loaded documents and embeddings (without text_splitter)
    vectordb=Chroma.from_documents(documents=glassnode_docs, embedding=embeddings , persist_directory=persist_path_full)
    # Persisting DB
    vectordb.persist()
else:
    print("Loading Existing DB!")
    # Load the existing persisted database from disk.
    vectordb = Chroma(persist_directory=persist_path_full, embedding_function=embeddings)


query = "Your query here"
docs_with_similarity_scores = vectordb.similarity_search_with_score(query)

for doc, score in docs_with_similarity_scores:
    # Print the first 300 characters of the document and its similarity score
    document_content = doc.page_content
    if len(document_content) > 300:
        document_content = document_content[:300] + "..."
    print(f"Document: {document_content}")
    print(f"Similarity Score: {score}\n")