# Service module. Not for deploy.

import chromadb

# chat_ollama_url_home: str = "http://192.168.1.57:11434"
chat_ollama_url_outdoor: str = "http://46.0.234.32:11434"

chroma_client = chromadb.HttpClient(host='46.0.234.32', port=8000)
# chroma_client = chromadb.HttpClient(host='192.168.1.57', port=8000)

print("Chroma current version: " + str(chroma_client.get_version()))
print("Collections count: " + str(chroma_client.count_collections()))
print("Chroma heartbeat: " + str(chroma_client.heartbeat()))

# print(chroma_client.list_collections(limit=100))
list_col = chroma_client.list_collections(100)
for col in list_col:
    # Преобразуем объект в строку и находим нужную часть
    name_part = str(col).split(", name=")[1].rstrip(")")
    print(name_part)

settings = chroma_client.get_settings()

for s in settings:
    print(s)

# Clear database:

chroma_client.reset()
chroma_client.clear_system_cache()
