# Search
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient

# os.environ["TAVILY_API_KEY"] = "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"

_ = load_dotenv()


def web_search(question: str, outputs_number: int = 1):
    """
        Не работает так же как и синхронная функция.
    Есть предположение, что выдает строку вместо списка словарей,
    где каждый словарь имеет ключ 'content'
    """
    tavily_client = TavilyClient(api_key="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")
    context = tavily_client.get_search_context(query=question, max_results=outputs_number, max_tokens=1000)
    print(context)

    return context
