import os
import aiofiles
import secrets
from quart import Quart, render_template, request, session, jsonify

from conv_async_graph_operator3 import Agent, run_agent  # Импортируем агента

app = Quart(__name__)

# Генерация случайного ключа
app.secret_key = secrets.token_hex(16)

# Создаем экземпляр агента один раз при запуске приложения
agent = Agent()

# Убедимся, что папка для загрузки файлов существует
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Названия коллекций
name_of_collection_1: str = "25_10_2024_LaBSE-en-ru_pdf"
name_of_collection_2: str = "23_10_2024_distiluse_txt"

# Асинхронная функция для общения с агентом
async def get_agent_response(user_text, agent, history):
    inputs = {
        "question": user_text,
        "history": history,
        "collection_name_1": name_of_collection_1,
        "collection_name_2": name_of_collection_2
    }
    result = await run_agent(agent, inputs)
    return result

@app.route("/")
async def home():
    return await render_template("index.html")

# Старый GET-метод, оставим как есть
@app.route("/get")
async def get_bot_response():
    user_text = request.args.get('msg')
    if 'history' not in session:
        session['history'] = []  # Если истории еще нет, создаем новый список

    session['history'].append({"role": "user", "content": user_text})
    response = await get_agent_response(user_text, agent, session['history'])
    session['history'].append({"role": "assistant", "content": response})
    return response

# Новый маршрут (API) для интеграции с Open Web UI или любым фронтендом
@app.route("/api/ask", methods=["POST"])
async def api_ask():
    # Предполагаем, что фронтенд отправляет JSON вида: {"question": "Привет?"}
    data = await request.get_json()
    question = data.get("question", "")

    if 'history' not in session:
        session['history'] = []

    session['history'].append({"role": "user", "content": question})
    response = await get_agent_response(question, agent, session['history'])
    session['history'].append({"role": "assistant", "content": response})

    # Возвращаем ответ в JSON-формате
    return jsonify({"answer": response})

@app.route("/upload", methods=["POST"])
async def upload_file():
    files = await request.files
    if 'file' not in files:
        return "No file part", 400

    file = files['file']
    filename = file.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    # Проверка на допустимые форматы
    if filename.endswith(".pdf") or filename.endswith(".txt"):
        # Асинхронное сохранение файла
        async with aiofiles.open(save_path, "wb") as f:
            await f.write(file.read())  # убрано await перед file.read()

        # Вызов функции для обработки и загрузки файла в Chroma DB
        process_file(save_path)
        return "File uploaded and processed successfully", 200
    else:
        return "Unsupported file type", 400

def process_file(file_path):
    """Функция для обработки файла и загрузки его в базу данных."""
    pass

if __name__ == "__main__":
    # Часто указывают host='0.0.0.0', чтобы сервер был виден извне.
    # Порт по умолчанию 5000. Можно поменять на 8000 при желании.
    app.run(debug=True, host="0.0.0.0", port=5000)