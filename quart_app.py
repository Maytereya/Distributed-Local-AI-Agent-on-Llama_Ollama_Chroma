# app.py
from quart import Quart, render_template, request, session
from conversational_async_graph_operator import Agent, run_agent  # Импортируем агента

app = Quart(__name__)
app.secret_key = '3380841777'  # Для обеспечения безопасности сессий

# Создаем экземпляр агента один раз при запуске приложения
agent = Agent()


# Асинхронная функция для общения с агентом

async def get_agent_response(user_text, agent, history):
    inputs = {"question": user_text, "history": history}

    # Запускаем асинхронную функцию агента для получения ответа
    result = await run_agent(agent, inputs)

    return result


@app.route("/")
async def home():
    return await render_template("index.html")


@app.route("/get")
async def get_bot_response():
    user_text = request.args.get('msg')

    # Инициализация или обновление истории в сессии
    if 'history' not in session:
        session['history'] = []  # Если истории еще нет, создаем новый список

    # Добавляем сообщение пользователя в историю
    session['history'].append({"role": "user", "content": user_text})

    # Получаем ответ от агента с текущей историей
    response = await get_agent_response(user_text, agent, session['history'])

    # Добавляем ответ бота в историю
    session['history'].append({"role": "assistant", "content": response})

    return response


if __name__ == "__main__":
    app.run(debug=True)
