import gradio as gr
from typing import Dict, Literal
from typing import List
from agent_logic_pack import aretrieve3 as retrieve


async def echo(message: str, history: List[Dict], slider_value: float, slider_value_n_results: int, slider_value_k,
               radio_value):
    return await retrieve.main_retrieve_async(question=message, return_type="str", threshold=slider_value,
                                              n_results=slider_value_n_results,
                                              k=slider_value_k,
                                              search_type=radio_value)


def radio_change(choice):
    if choice == "vectorstore":
        return gr.Slider(interactive=True), gr.Slider(interactive=True), gr.Slider(interactive=False),
    else:
        return gr.Slider(interactive=False), gr.Slider(interactive=False), gr.Slider(interactive=True),


with gr.Blocks() as blocks:
    gr.Markdown("# NEIRY.AI **bookworm**")
    chatbot = gr.Chatbot(type="messages", autoscroll=True,
                         placeholder="<strong>Поиск по документам</strong><br>Задайте вопрос")
    textbox = gr.Textbox(lines=1, placeholder="напишите вопрос", submit_btn=True, container=True, autoscroll=True,
                         autofocus=True)

    with gr.Column():
        radio = gr.Radio(["vectorstore", "db", ],
                         label="Способ первичного поиска", value="vectorstore", container=True)

    with gr.Row():
        slider3 = gr.Slider(value=5, minimum=1, maximum=20, step=1,
                            label="Количество чанков текста, включенных в выдачу, vectorstore - поиск",
                            info="Только в режиме поиска vectorstore",
                            interactive=True,
                            )

        slider1 = gr.Slider(value=0.005, minimum=0.0025, maximum=0.02, step=0.0025,
                            label="Пороговое значение косинусной фильтрации",
                            info="Выбрать в диапазоне между 0.0025 и 0.02. "
                                 "Чем больше значение, тем больше чанков с меньшей "
                                 "релевантностью будет в выдаче",
                            interactive=True
                            )

        slider2 = gr.Slider(value=2, minimum=1, maximum=20, step=1,
                            label="Количество документов, включенных в выдачу, db - поиск",
                            info="Только в режиме поиска db",
                            interactive=False,
                            )
    with gr.Column():
        demo = gr.ChatInterface(fn=echo, type="messages",
                                examples=[["апатия, причины, лечение"], ["ангедония, причины, лечение"],
                                          ["акатизия, причины, лечение"], ["ЗНС, лечение"]],
                                # title="MMR - поиск в тексте с косинусной фильтрацией",
                                chatbot=chatbot,
                                textbox=textbox,
                                additional_inputs=[slider1,
                                                   slider2,
                                                   slider3,
                                                   radio,
                                                   ],
                                show_progress="full")

        radio.change(fn=radio_change, inputs=radio, outputs=[slider3, slider1, slider2])

if __name__ == "__main__":
    blocks.launch()
