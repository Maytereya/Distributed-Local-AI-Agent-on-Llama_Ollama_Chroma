# This is a sample Python script.
from pprint import pprint

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import hybrid


agent = hybrid.Agent()
app = agent.graph
inputs = {"question": "Why the sky is blue?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
