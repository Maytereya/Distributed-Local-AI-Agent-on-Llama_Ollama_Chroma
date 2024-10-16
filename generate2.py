from langchain_core.documents import Document

import config as c  # Here are all ip, llm names and other important things
import time
from ollama import AsyncClient
import json_converter as j

ollama_aclient = AsyncClient(host=c.ollama_url)

# Выбор llm
llm = c.ll_model_big


# Post-processing
def format_docs(docs):
    """Convert Document to string
        This option need for some functions inside async_graph_operator.py: generate_final
    """
    return "\n\n".join(doc.page_content for doc in docs)


async def chat(question: str, history=None):
    """
Just chat with the short-term history
    """
    # if history is None:
    #     history = [{"role": "user", "content": "no history"}]
    if history is None:
        # Память!
        history = []
    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for '
              'question-answering tasks. Answer the question in '
              'plain text format. If you do not know the answer, just say that you do not know. Use three sentences '
              'maximum and keep the answer concise. History of previous conversations should be referenced, '
              f'if applicable, and included in the answer. <|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Question: {question}. History of previous conversations: {history} Answer: '
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=c.ll_model,
        prompt=prompt,
        # format="json",
        # options=opt,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")
    #

    return aresult['response']


async def generate_answer(question: str, documents: list[Document], history=None) -> list[Document]:
    """
    Generate the answer if the agent
    Пока рекордсмен по продолжительности генерации...
    """
    # if history is None:
    #     history = [{"role": "user", "content": "no history"}]
    if history is None:
        history = []
    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for '
              'question-answering tasks. Use the following pieces of retrieved context to answer the question in '
              'plain text format. If you do not know the answer, just say that you do not know. Use three sentences '
              'maximum and keep the answer concise. History of previous conversations should be referenced, '
              f'if applicable, and included in the answer.  <|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Question: {question}. Context: {documents}. History here: {history} Answer: '
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=c.ll_model,
        prompt=prompt,
        # format="json",
        # options=opt,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")
    #
    print("aresult['response'] type is: ", type(aresult))
    return aresult['response']
