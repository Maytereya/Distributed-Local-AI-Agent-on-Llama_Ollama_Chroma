# Search
import asyncio
import os

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults


# os.environ["TAVILY_API_KEY"] = "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"


# _ = load_dotenv()


async def aweb_search(question: str, outputs_number: int = 2):
    """
        Не работает так же как и синхронная функция.
    Есть предположение, что выдает строку вместо списка словарей,
    где каждый словарь имеет ключ 'content'
    """
    web_search_tool = TavilySearchResults(k=outputs_number)
    search_result = await web_search_tool.ainvoke({"query": question})
    return search_result


def web_search(question: str, outputs_number: int = 2):
    web_search_tool = TavilySearchResults(k=outputs_number)
    return web_search_tool.invoke({"query": question})

# result = asyncio.run(web_search("Ollama Chat"))
#
# print(result)
# print(type(result))



# import json
# from typing import Dict, Any
#
# def web_search(self, state: AgentState):
#     """
#     Эта функция выполняет веб-поиск на основе вопроса и добавляет результаты к документам.
#
#     Args:
#         state (dict): The current graph state
#
#     Returns:
#         state (dict): Appended web results to documents
#     """
#
#     print("---TAVILY WEB SEARCH---")
#     question = state["question"]
#     documents = state["documents"]
#
#     # Web search
#     docs_str: str = asyncio.run(search_01.web_search(question))  # Предположим, что возвращается JSON-строка
#     try:
#         docs: List[Dict[str, Any]] = json.loads(docs_str)  # Преобразуем строку в список словарей
#         combined_results = "\n".join([d["content"] for d in docs])
#     except (json.JSONDecodeError, TypeError, KeyError) as e:
#         print(f"Failed to process search results: {e}")
#         combined_results = ""
#
#     web_results = Document(page_content=combined_results)
#     if documents is not None:
#         documents.append(web_results)
#     else:
#         documents = [web_results]
#
#     return {"documents": documents, "question": question}
