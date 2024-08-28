from langchain.chains.question_answering.map_reduce_prompt import messages
from sqlalchemy.engine.reflection import cache
from torch.cuda import temperature

from retrieve_03 import QueryCollection as Qc
import config as c
import asyncio
import time
import routing_01 as r

from ollama import AsyncClient, Client, Options, Message
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


# Retrieving test
def rt():
    start_time = time.time()

    qc = Qc(ollama_url=c.ollama_url, chroma_host=c.chroma_host, chroma_port=c.chroma_port,
            embedding_model=c.emb_model)

    documents = asyncio.run(qc.async_launcher(question=c.question1, collection_name=c.collect_name))

    # Замеряем время окончания выполнения asyncio.run
    end_time = time.time()

    # Рассчитываем общее время выполнения
    elapsed_time = end_time - start_time

    print("---RETRIEVE FROM CHROMA---")
    for d in documents:
        print(d)

    # Выводим время выполнения в секундах
    print(f"Время выполнения скрипта: {elapsed_time:.2f} секунд")


rt()

# Резюме: через ВПН вообще не работает.
print("Время выполнения скрипта: 2.31 секунд (Нейро-Пси Мск)")


# Routing test
def ro_t():
    start_time = time.time()
    result = asyncio.run(r.route(c.question1))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(result)
    # Выводим время выполнения в секундах
    print(f"Время выполнения скрипта: {elapsed_time:.2f} секунд")


# ro_t()

# Резюме:
#  Тайминг: Время выполнения скрипта: 43.54 секунд
# Не приемлемо вообще.

# Пытаемся собрать ту же херхню быстрым способом без выяснения причин того почему ЧатОллама такой медленный


async def route(question: str):
    """Routing Questions to range of collections or WebSearch"""

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, 
        prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return the JSON with a single key 'datasource' and 
        no preamble or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = prompt | c.llm | JsonOutputParser()
    return await question_router.ainvoke({"question": question})


async def try_client():
    ollama_aclient = AsyncClient(host=c.ollama_url)
    ollama_client = Client(host=c.ollama_url)
    opt = Options(temperature=0, num_gpu=2, num_thread=24)

    question = c.question1
    prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user "
              f"question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, "
              f"prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords in the "
              f"question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' or "
              f"'vectorstore' based on the question. Return the JSON with a single key 'datasource' and no preamble or "
              f"explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>")

    msg = Message(role='user', content=prompt, )

    # async & .generate
    print("async & .generate")
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=c.ll_model,
        prompt=prompt,
        # format="json",
        # options=opt,
        keep_alive=-1,

    )

    print(aresult['response'])
    print(f"eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Время выполнения асинхронного запроса к клиенту: {elapsed_time:.2f} секунд")
    print('Время выполнения асинхронного запроса к клиенту: 3.56 секунд (LTE, MSK)')

    # normal & .generate
    print("normal & .generate")
    start_time = time.time()
    result = ollama_client.generate(
        model=c.ll_model,
        prompt=prompt,
        format="json",

    )

    print(result['response'])

    print(f"eval_duration: {result['eval_duration'] / 1_000_000_000}")
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Время выполнения синхронного запроса к клиенту: {elapsed_time:.2f} секунд")
    print('Время выполнения синхронного запроса к клиенту: 4.67 секунд (LTE, MSK)')

    # async & .chat function
    print('async & .chat')
    start_time = time.time()
    ames = await ollama_aclient.chat(
        model=c.ll_model,
        messages=[msg],
        format="json",

    )

    content_value = ames['message']['content']
    eval_duration_value = ames['eval_duration']

    print(f"content: {content_value}")
    print(f"eval_duration: {eval_duration_value / 1_000_000_000}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Время выполнения асинхронного запроса к клиенту .chat  : {elapsed_time:.2f} секунд")

    # normal & .chat function
    print('normal & .chat')
    start_time = time.time()
    mes = ollama_client.chat(
        model=c.ll_model,
        messages=[msg],
        format="json",

    )

    content_value = mes['message']['content']
    eval_duration_value = mes['eval_duration']

    print(f"content: {content_value}")
    print(f"eval_duration: {eval_duration_value / 1_000_000_000}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Время выполнения синхронного запроса к клиенту .chat  : {elapsed_time:.2f} секунд")


asyncio.run(try_client())
# Время выполнения асинхронного запроса к клиенту: 5.85 секунд (LTE)
# Время выполнения синхронного запроса к клиенту: 4.65 секунд (LTE)
# Использование параметра options "сажает" время ответа до 400+ секунд.
