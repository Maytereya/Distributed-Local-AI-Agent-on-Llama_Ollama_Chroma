import chromadb
import ollama
# from langchain.embeddings import ollama
from ollama import Client
from langchain_ollama import OllamaEmbeddings

# chat_ollama_url_home: str = "http://192.168.1.57:11434"
chat_ollama_url_outdoor: str = "http://46.0.234.32:11434"
# import index

chroma_client = chromadb.HttpClient(host='46.0.234.32', port=8000)
# chroma_client = chromadb.HttpClient(host='192.168.1.57', port=8000)

print(chroma_client.get_version())
print(chroma_client.count_collections())

print(chroma_client.heartbeat())
# index.create_collection(chroma_client)

# print(chroma_client.list_collections(limit=100))
list_col = chroma_client.list_collections(100)
for col in list_col:
    # Преобразуем объект в строку и находим нужную часть
    name_part = str(col).split(", name=")[1].rstrip(")")
    print(name_part)



settings = chroma_client.get_settings()

for s in settings:
    print(s)

# chroma_client.reset()
# chroma_client.clear_system_cache()


# # Чисто для проверки работоспособности
# client = Client(host=chat_ollama_url_home)
# response = client.chat(model='llama3', messages=[
#     {
#         'role': 'user',
#         'content': 'Why sky is blue?',
#     },
# ])
# print(response)

# to_print = client.embeddings(model="mxbai-embed-large", prompt="Llamas are members of the camelid family meaning
# they're pretty closely related to vicuñas and camels")
#
# print(to_print)
