import json
from typing import Dict, Optional
import re


def str_to_json(input_str: str) -> str:
    try:
        # Попытка сразу загрузить строку в формате JSON
        return json.loads(input_str)
    except json.JSONDecodeError:
        # Обработка распространенных ошибок формата

        # Удаление лишних символов, таких как пробелы и переносы строк
        input_str = input_str.strip()

        # Замена одинарных кавычек на двойные
        input_str = input_str.replace("'", '"')

        # Удаление обратных слешей перед кавычками, если такие есть
        input_str = re.sub(r'\\(.)', r'\1', input_str)

        # Повторная попытка загрузить JSON после исправлений
        try:
            return json.loads(input_str)
        except json.JSONDecodeError as e:
            # Если строку все равно не удалось преобразовать, выводим сообщение об ошибке
            print(f"Ошибка в конвертере: не получается преобразование строки в JSON: {e}")
            return None
