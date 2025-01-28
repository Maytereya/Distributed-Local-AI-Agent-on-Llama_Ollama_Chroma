from langgraph.graph import START, END
import config as c
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from ollama import Client, Message
from langchain_community.tools.tavily_search import TavilySearchResults
import routing_for_chat as r
from langgraph.checkpoint.memory import MemorySaver

_ = load_dotenv()


def add_ollama_messages(
        left: list[Message] | Message,
        right: list[Message] | Message
) -> list[Message]:
    """
    Функция - редьюсер для добавления нового сообщения в список сообщений,
    вместо того чтобы заменять его.
    Содержит проверку типа данных.
    Не содержит проверку ID и поведение в зависимости от этого.
    Message не содержит поле ID.
    """
    # Проверяем и преобразуем в списки
    if not isinstance(left, list):
        left = [left]

    if not isinstance(right, list):
        right = [right]

    # Объединяем сообщения
    return left + right


def convert_to_base_message(message: Message) -> BaseMessage:
    """
    Takes ollama Message type and convert to BaseMessage
    Uses when need to use add_messages reducer function.

    :param message: The ollama Message to convert
    :return: BaseMessage or its subclass
    """
    role = message['role']
    content = message['content']

    # Проверяем, есть ли tool_calls
    if 'tool_calls' in message and message['tool_calls']:
        # Предполагаем, что это список объектов ToolCall, и создаем ToolMessage
        tool_calls = message['tool_calls']
        # Возвращаем ToolMessage
        return ToolMessage(content=content, tool_calls=tool_calls)

    # Конвертация стандартных сообщений
    if role == 'user':
        return HumanMessage(content=content)
    elif role == 'assistant':
        return AIMessage(content=content)
    else:
        # Обработка других ролей или использование BaseMessage напрямую
        return BaseMessage(role=role, content=content)


memory = MemorySaver()


class State(TypedDict):
    # messages: Annotated[list[AnyMessage], operator.add]
    # messages: Annotated[list[Message], add_ollama_messages]  # Используем редьюсер ollama для сообщений
    # messages: Annotated[list[Message], add_messages]
    messages: Message
    # messages: list[Message]  # Используем тип ollama для сообщений 2 вариант
    # messages: Annotated[list, add_messages]
    # question: str
    # generation: str
    # web_search: str
    tools: list
    request: str
    # documents: List[str]  #Возможно ошибочно
    # documents: Optional[List[Document]]  # С поправкой на ошибку несоответствия типа в модуле web_search


graph_builder = StateGraph(State)

tool = TavilySearchResults(max_results=2)
o_client = Client(c.ollama_url)

# Пример сообщений и инструментов
messages = [Message(role='user', content='Какая сегодня погода в Москве?')]

tools = [{
    'type': 'function',
    'function': {
        'name': 'get_current_weather',
        'description': 'Get the current weather for a city',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'The name of the city',
                },
            },
            'required': ['city'],
        },
    }
}]


# Входная функция !
# async def route_question(state: State):
#     """
#     Эта функция определяет, куда направить вопрос: на веб-поиск или векторное хранилище.
#
#     Args:
#         state (dict): The current graph state
#
#     Returns:
#         str: Next node to call
#     """
#
#     print("---ROUTE QUESTION---")
#     question = state["question"]
#     print(question)
#
#     source = await route(question)
#
#     if source["datasource"] == "web_search":
#         print("---ROUTE QUESTION TO WEB SEARCH---")
#         return "websearch"
#     elif source["datasource"] == "vectorstore":
#         print("---ROUTE QUESTION TO RAG---")
#         return "vectorstore"


# async def route(question: str):
#     ollama_aclient = AsyncClient(host=c.ollama_url)
#     # options make Ollama slow so far.
#     # opt = Options(temperature=0, num_gpu=2, num_thread=24)
#
#     prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user '
#               'question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, '
#               'prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords in the '
#               'question related to these topics. Otherwise, use web-search. Give a binary choice "web_search" or '
#               '"vectorstore" based on the question. Return the JSON with a single key "datasource" and no preamble or '
#               'explanation. Example#1: {"datasource": "web_search"}, example#2: {"datasource": "vectorstore"}. '
#               'Attention: if you want to make a choice "web_search", think twice! Maybe its a mistake.'
#               f'Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>')
#
#     # async & .generate
#     start_time = time.time()
#     aresult = await ollama_aclient.generate(
#         model=c.ll_model,
#         prompt=prompt,
#         # format="json",
#         # options=opt,
#         keep_alive=-1,
#
#     )
#
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     # Без await не работают!
#     print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
#     print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")
#
#     json_result = j.str_to_json(aresult['response'])  # Carefully check the format
#     print("Router response: " + str(json_result))
#     return json_result


# def web_search(question: str, outputs_number: int = 1):
#     """
#     """
#     tavily_client = TavilyClient()
#     start_time = time.time()
#     context = tavily_client.get_search_context(query=question, max_results=outputs_number, max_tokens=1000)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(
#         f"Tavily research timing: {elapsed_time:.2f} sec")
#     return context


# llm = ChatOllama(model=c.ll_model_big, base_url=c.ollama_url)


def route_question(state: State):
    """
    Эта функция определяет, куда направить вопрос: на веб-поиск или векторное хранилище.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["messages"][0]["content"]
    print(question)

    source = r.route(question)

    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "continue_chatting":
        print("---ROUTE QUESTION TO Continue Chatting---")
        return "continue_chatting"


# def chatbot(state: State):
#     return {"messages": [o_client.chat(model=c.ll_model, messages=state["messages"])]}

# def chatbot(state: State):
#     return {"messages": [llm.invoke(state["messages"])]}


# state = State(messages=messages, tools=tools, request="test_request")


def chatbot(state: State):
    # Обозначаем модель и ее URL - адрес:
    print(f"Model: {c.ll_model_big}")
    print(f"Current ollama server URL: {c.ollama_url}")

    # Первая попытка вызвать .chat
    response = o_client.chat(
        model=c.ll_model_big,
        messages=state['messages'],
        # provide a weather checking tool to the model
        tools=state['tools'],
    )
    # Распечатываем содержимое важной переменной response
    print("eval_duration: " + str(response['eval_duration'] / 1_000_000_000))
    # print("response group")
    # print(response['message']['tool_calls'][0])
    # print(response['message']['tool_calls'][0]['function']['name'])
    # print(response['message']['tool_calls'][0]['function']['arguments']['city'])

    # Вариант подключения инструмента
    if response['message']['tool_calls'][0]['function']['name'] == 'get_current_weather':
        city = response['message']['tool_calls'][0]['function']['arguments']['city']
        state['request'] = f"Погода в городе {city} сейчас"
        # weather = tool.invoke(state['request'])

    # return response['message']
    return convert_to_base_message(response['message'])


# print("Вызов функции: " + str(try_response(state)))
#
# print("state after function call group")
# print(state['request'])
# print(state['messages'])

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
# graph = graph_builder.compile()
# graph = graph_builder.compile(checkpointer=memory)
graph = graph_builder.compile()
# config = {"configurable": {"thread_id": "1"}}

# The config is the **second positional argument** to stream() or invoke()!
user_input = "Hi there! My name is Will."

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {messages : Message(role='user', content='Какая сегодня погода в Москве?')}
    # {"messages": [("user", user_input)]}
)

for event in events:
    # event["messages"][-1].pretty_print()
    print(event)
#     # print(event['messages'][1].content)

# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye!")
#         break
#     for event in graph.stream({"messages": ("user", user_input)}):
#         # print(event)
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)
