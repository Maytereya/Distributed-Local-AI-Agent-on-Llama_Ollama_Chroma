import config as c  # Here are all ip, llm names and other important things
from ollama import AsyncClient
import asyncio

ollama_aclient = AsyncClient(host=c.ollama_url)


async def check(document, claim):
    """Checks if the claim is supported by the document by calling bespoke-minicheck.

    Returns Yes/yes if the claim is supported by the document, No/no otherwise.
    Support for logits will be added in the future.

    bespoke-minicheck's system prompt is defined as:
      'Determine whether the provided claim is consistent with the corresponding
      document. Consistency in this context implies that all information presented in the claim
      is substantiated by the document. If not, it should be considered inconsistent. Please
      assess the claim's consistency with the document by responding with either "Yes" or "No".'

    bespoke-minicheck's user prompt is defined as:
      "Document: {document}\nClaim: {claim}"
    """

    system_prompt = "Determine whether the provided claim is consistent with the corresponding document. "
    "Consistency in this context implies that all information presented in the claim is substantiated by the document. "
    "If not, it should be considered inconsistent. Please assess the claim's consistency "
    "with the document by responding with either 'yes' or 'no'."

    prompt = f"Document: {document}\nClaim: {claim}"

    response = await ollama_aclient.generate(
        model="bespoke-minicheck", prompt=prompt, options={"num_predict": 2, "temperature": 0.0},
        system=system_prompt,
    )
    # if response["response"].strip().lower() == "си":
    #     response = "yes"

    return response


doc = """ Розы красные """

ques = "Розы голубые"

if __name__ == "__main__":
    print(asyncio.run(check(doc, ques)))
