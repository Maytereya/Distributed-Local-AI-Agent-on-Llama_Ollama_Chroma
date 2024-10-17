import config as c  # Here are all ip, llm names and other important things
import time
from ollama import AsyncClient
import json_converter as j

ollama_aclient = AsyncClient(host=c.ollama_url)
# Выбор llm
llm = c.ll_model_big


async def grade(question: str, document: str):
    """
    Grade the relevance of the retrieved document.
    """
    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'You are tasked with assessing the relevance of a retrieved document to a user’s question. '
              'If the document contains concepts or keywords closely related to the user’s question, consider it relevant. '
              'The assessment does not need to be overly strict—the aim is to filter out irrelevant documents, not to demand exact matches. '
              'Return a binary "yes" or "no" to indicate whether the document is relevant. '
              'Provide your answer as a JSON object with a single key "score" and no additional text. '
              'Example: {"score": "yes"} or {"score": "no"}.'
              '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Here is the retrieved document: \n\n{document} \n\n'
              f'Here is the user question: {question} \n\n'
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        # format="json",
        # options=opt,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")
    #
    json_result = j.str_to_json(aresult['response'])
    print("Grade retrieved response: " + str(json_result))

    return json_result


async def hallucinations_checker(documents, generation):
    """Hallucination Grader"""

    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'You are tasked with evaluating whether a generated answer is fully grounded in and supported by the provided facts. '
              'Your goal is to determine if the answer directly relies on these facts, without introducing unsupported information. '
              'Give a binary "yes" or "no" score to indicate whether the answer is factually supported. '
              'Return the score as a JSON object with a single key "score", and do not provide any explanation or extra text. '
              'Example: {"score": "yes"} or {"score": "no"}. '
              '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Here are the provided facts: \n\n{documents} \n\n'
              f'Here is the generated answer: \n\n{generation} \n\n'
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
    # print('Предыдущий результат: 227.32 секунд (LTE, MSK)')
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])
    print("Hallucinations checker response: " + str(json_result))
    return json_result


async def answer_grader(question: str, generation):
    """Answer Grader"""

    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'You are tasked with evaluating whether a generated answer is useful for resolving the user’s question. '
              'Your goal is to determine if the answer directly addresses the question and provides meaningful or helpful information. '
              'Assign a binary score: "yes" if the answer is useful, or "no" if it is not. '
              'Provide the score as a JSON object with a single key "score", without any additional explanation or preamble. '
              'Example: {"score": "yes"} or {"score": "no"}. '
              '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Here is the generated answer: \n\n{generation} \n\n'
              f'Here is the question: \n\n{question} \n\n'
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
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])
    print("Answer grader response: " + str(json_result))

    return json_result
