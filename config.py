from langchain_ollama import ChatOllama
from langchain_community.cache import InMemoryCache

# Input variables:
collect_name: str = "rag-ollama"
ollama_url_out: str = "http://46.0.234.32:11434"
ollama_url_in: str = "http://192.168.1.57:11434"
chat_ollama_url_belgium: str = "http://46.183.187.205:11434"
ollama_url = ollama_url_out  # Итоговое значение

chroma_host_in: str = "192.168.1.57"
chroma_host_out: str = "46.0.234.32"
chroma_host = chroma_host_out  # Итоговое значение

chroma_port = 8000

# emb_model = "llama3.1:8b-instruct-fp16"
emb_model = "llama3.1:latest"
# ll_model = "llama3.1:8b-instruct-fp16"
ll_model = "llama3.1:latest"
# question1 = "What is agent memory?"
# local_llm = "llama3.1:8b-instruct-fp16"


global_cache = InMemoryCache()

llm = ChatOllama(model=ll_model, base_url=ollama_url, cache=True, keep_alive=-1, num_gpu=2, num_thread=24,
                 num_predict=-1,
                 format="json",
                 temperature=0)
llm.cache = global_cache

# urls_rus = [
#     "https://neiro-psy.ru/blog/10-sovetov-dlya-poiska-lyubvi-i-znakomstv-pri-socialnoj-trevoge",
# ]
#
# urls_reserve = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
# ]

# self.urls = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
# ]
