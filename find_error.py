from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

import ollama

import uuid

import chromadb
from chromadb.config import Settings
import os

urls_rus = [
    "https://vc.ru/crypto/1373903-kak-passivno-zarabatyvat-na-kriptovalyute",
]
# Variables group for chroma collections management
query_chroma_test: str = "What is the Adversarial attacks?"
query_chroma_test_rus: str = "Как начать стейкинг?"
collection_name: str = "rag-rus-ollama2"

persist_directory = "/Users/rakhmanov/Desktop/#dev/chroma_db"
chat_ollama_url_home: str = "http://192.168.1.57:11434"
chat_ollama_url_outdoor: str = "http://46.0.234.32:11434"
chat_ollama_url_belgium: str = "http://46.183.187.205:11434"

chroma_client_host_home: str = "192.168.1.57"
chroma_client_host_outdoor: str = "46.0.234.32"
chroma_client_port: int = 8000
local_llm: str = "llama3"


def txt_splitter(add_urls: List[str]) -> List[Document]:
    docs = [WebBaseLoader(url).load() for url in add_urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print("Text Splitting Done.")
    return doc_splits


# Теперь чисто для использования ollama Embeddings

chroma_client = chromadb.HttpClient(host=chroma_client_host_outdoor, port=chroma_client_port)

ollama_collection = chroma_client.create_collection(name=collection_name)
print("Создаем коллекцию")
# store each document in a vector embedding database with Ollama
docs = txt_splitter(urls_rus)  # Подставляем правильную переменную в зависимости от источника данных.
print("Сплитим доки")
for doc in docs:
    print("Инициализируем переменную response")
    response = ollama.embeddings(model="mxbai-embed-large", prompt=doc)
    embedding = response["embedding"]
    ollama_collection.add(
        ids=[str(uuid.uuid1())], embeddings=[embedding], metadatas=doc.metadata, documents=doc.page_content
    )

# an example prompt
prompt = query_chroma_test_rus

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
    prompt=prompt,
    model="mxbai-embed-large"
)
results = ollama_collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)
data = results['documents'][0][0]
print("Ответ Ollama Embeddings:")
print(data)
