from transformers import pipeline
import json
import re


class JSONExtractionBot:
    def __init__(self):
        print("Загрузка модели для извлечения данных...")
        self.generator = pipeline(
            "text-generation",
            model="inkoziev/rugpt_chitchat",
            device_map="auto"
        )
        print("Модель загружена!")

    def extract_inn_and_name(self, text):
        """Специализированный метод для извлечения ИНН и ФИО"""
        prompt = f"""
        ТЕКСТ: "{text}"

        ИНСТРУКЦИЯ: Извлеки из текста ИНН и ФИО клиента. 
        ИНН должен состоять только из цифр (10 или 12 символов).
        ФИО должно содержать фамилию, имя и отчество.

        ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON:
        {{
            "ИНН": "найденный_инн",
            "ФИО": "полное_фио"
        }}

        ПРИМЕР:
        Текст: "ИНН 1234567890, Петров Алексей Сергеевич"
        Ответ: {{"ИНН": "1234567890", "ФИО": "Петров Алексей Сергеевич"}}

        Ответ для данного текста:
        """

        response = self.generator(
            prompt,
            max_new_tokens=150,
            temperature=0.1,  # Низкая температура для точности
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )[0]['generated_text']

        return self._extract_and_validate_json(response, text)

    def extract_custom_data(self, text, fields):
        """Универсальный метод для извлечения любых данных"""
        fields_str = ", ".join([f'"{field}"' for field in fields])

        prompt = f"""
        ИНСТРУКЦИЯ: Извлеки из текста указанные поля и верни в формате JSON.
        Текст: "{text}"
        Поля для извлечения: [{fields_str}]

        Формат ответа:
        {{
            {', '.join([f'"{field}": ""' for field in fields])}
        }}

        Ответ только в JSON:
        """

        response = self.generator(
            prompt,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            num_return_sequences=1
        )[0]['generated_text']

        return self._extract_and_validate_json(response, text)

    def _extract_and_validate_json(self, response, original_text):
        """Извлекает JSON и валидирует данные"""
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{[^}]*\}', response)
            if not json_match:
                return {
                    "error": "JSON не найден в ответе",
                    "raw_response": response,
                    "original_text": original_text
                }

            json_str = json_match.group()
            data = json.loads(json_str)

            # Валидация ИНН если есть
            if "ИНН" in data:
                inn = str(data["ИНН"]).strip()
                # Оставляем только цифры
                inn_digits = re.sub(r'\D', '', inn)
                if len(inn_digits) in [10, 12]:
                    data["ИНН"] = inn_digits
                else:
                    data["ИНН_валидация"] = "Неверная длина ИНН"

            return data

        except json.JSONDecodeError as e:
            return {
                "error": f"Ошибка парсинга JSON: {e}",
                "raw_response": response,
                "original_text": original_text
            }


def main():
    bot = JSONExtractionBot()

    print("🤖 Бот для извлечения данных в JSON")
    print("Команды: 'выход' - завершить, 'пример' - показать пример")
    print("-" * 50)

    while True:
        user_input = input("\nВведите текст для анализа: ").strip()

        if user_input.lower() in ['выход', 'exit', 'quit']:
            break
        elif user_input.lower() == 'пример':
            print("\nПримеры текстов:")
            print('1. "Аккр n 123, Инн 4353229845, Иванов Иван Иванович"')
            print('2. "Клиент: Сидоров Петр Васильевич, ИНН 123456789012"')
            print('3. "ФИО: Козлова Мария Сергеевна, инн 9876543210"')
            continue

        if not user_input:
            continue

        print("\n🔍 Анализирую текст...")
        result = bot.extract_inn_and_name(user_input)

        print("\n📊 Результат:")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()