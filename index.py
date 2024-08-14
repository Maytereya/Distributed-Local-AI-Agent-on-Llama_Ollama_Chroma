# hybrid index v 1.0

from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_nomic.embeddings import NomicEmbeddings
from chromadb.utils import embedding_functions  # Пытаемся установить all-MiniLM-L6-v2, который по умолчанию в Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
from ollama import Client
import ollama

import uuid

import chromadb
from chromadb.config import Settings
import os

# Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

# Index

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
urls_rus = [
    "https://neiro-psy.ru/blog/10-sovetov-dlya-poiska-lyubvi-i-znakomstv-pri-socialnoj-trevoge",
]
# Variables group for chroma collections management
query_chroma_test: str = "What is the Adversarial attacks?"
query_chroma_test_rus: str = "Что такое социальная тревога?"
#
collection_name: str = "rag-rus-ollama4"
#
embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
# embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings(model_name="bert-base-multilingual-cased")

persist_directory = "/Users/rakhmanov/Desktop/#dev/chroma_db"
chat_ollama_url_home: str = "http://192.168.1.57:11434"
chat_ollama_url_outdoor: str = "http://46.0.234.32:11434"
chat_ollama_url_belgium: str = "http://46.183.187.205:11434"
default_ef = embedding_functions.DefaultEmbeddingFunction()  # Hope that it is all-MiniLM-L6-v2
# Suppose that it's not ((( Dimensions not matched*
chroma_client_host_home: str = "192.168.1.57"
chroma_client_host_outdoor: str = "46.0.234.32"
chroma_client_port: int = 8000
local_llm: str = "llama3"


def txt_splitter(add_urls: List[str]) -> List[Document]:
    docs = [WebBaseLoader(url).load() for url in add_urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=30
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print("Text Splitting Done.")
    return doc_splits


def add_to_chromadb():
    """Add to vectorDB"""
    doc_splits = txt_splitter(urls)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
    )
    retriever = vectorstore.as_retriever()
    print("Adding to ChromaDB Done.")
    return retriever


# Не ясно как обратиться к ретриверу без очередного добавления документов в базу.
# Ответ прост: это все ин-мемори. Он никуда ничего не сохраняет.
def add_to_chroma_disk():
    """save to disk"""
    doc_splits = txt_splitter(urls)

    db2 = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        persist_directory=persist_directory
    )
    print(f"Коллекция {collection_name} сохранена на диск на этом маке")
    output = db2.similarity_search(query_chroma_test)
    return output


def load_from_chroma_disk():
    """load from disk"""
    db3 = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    print(f"Коллекция {collection_name} загружена с диска этого мака")
    output = db3.similarity_search(query_chroma_test)
    print(output[0].page_content)
    return output


def create_chroma_client():
    """create the chroma client"""
    # client = chromadb.HttpClient(host=chroma_client_host_home, port=chroma_client_port,
    #                              settings=Settings(allow_reset=True))
    client_ = chromadb.HttpClient(host=chroma_client_host_home, port=chroma_client_port)
    return client_


def create_collection(client):
    # client.reset()  # resets the database
    docs_ = txt_splitter(urls_rus)  # Подставляем правильную переменную в зависимости от источника данных.
    collection = client.create_collection(collection_name)
    for doc_ in docs_:
        collection.add(
            ids=[str(uuid.uuid1())], metadatas=doc_.metadata, documents=doc_.page_content
        )


# tell LangChain to use our client and collection name
def make_a_query(query):
    db4 = Chroma(
        client=create_chroma_client(),
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    output = db4.similarity_search(query)
    print(output[0].page_content)


# ollama Embeddings
# ollama.embeddings(model='llama3.1', prompt='The sky is blue because of rayleigh scattering') # Пример функции

emb_model = "llama3.1:8b-instruct-fp16"
# emb_model = "mxbai-embed-large"
# emb_model = "nextfire/paraphrase-multilingual-minilm"
ll_model = "llama3.1:8b-instruct-fp16"

chroma_client = chromadb.HttpClient(host=chroma_client_host_home, port=chroma_client_port)
client = Client(host=chat_ollama_url_home)
ollama_collection = chroma_client.create_collection(name=collection_name)
print(f"Создаем коллекцию (STEP 1): {collection_name}")

# STEP 1 :: store each document in a vector embedding database with Ollama
docs = txt_splitter(urls_rus)  # Подставляем правильную переменную в зависимости от источника данных.

i = 0
for doc in docs:
    i += 1
    print(f"Цикл переменной response № {i}")
    response = client.embeddings(model=emb_model, prompt=doc.page_content)
    # response = ollama.embeddings(model=model, prompt=doc.page_content)
    embedding = response["embedding"]
    ollama_collection.add(
        ids=[str(uuid.uuid1())], embeddings=[embedding], metadatas=doc.metadata, documents=doc.page_content
    )

# an example prompt
prompt = query_chroma_test_rus

# STEP 2 :: generate an embedding for the prompt and retrieve the most relevant doc
# response = ollama.embeddings(
response = client.embeddings(
    prompt=prompt,
    model=emb_model
)
results = ollama_collection.query(
    query_embeddings=[response["embedding"]],
    n_results=5
)
data = results['documents'][0][0] + "\n" + results['documents'][0][1] + "\n" + results['documents'][0][2] + "\n" + \
       results['documents'][0][3] + "\n" + results['documents'][0][4]
print(f"Ответ Ollama Embeddings {emb_model} на вопрос {prompt} (STEP 2):")
print(data)

# STEP 3 :: generate a response combining the prompt and data we retrieved in step 2 (above)

output = client.generate(
    model=ll_model,
    prompt=f"Используя данные из : {data} ответь на этот вопрос на русском языке: {prompt}. Если данные {data} не "
           f"содержат ответа на вопрос {prompt}, ответь 'данные не подходят' без дальнейших объяснений"
)
print(f"Ответ модели {ll_model} на вопрос {prompt} (STEP 3):")
print(output['response'])

# Retrieval Grader

llm = ChatOllama(model=local_llm, base_url=chat_ollama_url_outdoor, format="json", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()

question = query_chroma_test
docs = add_to_chromadb().invoke(question)
doc_txt = docs[1].page_content
print("======================")
print(f"Квалификация вопроса: {question}")
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
print("========================")
