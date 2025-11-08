from transformers import pipeline
import json
import re


class ForceModelExtractor:
    def __init__(self):
        print("🚀 Загрузка модели с принудительным форматом...")
        self.generator = pipeline(
            "text-generation",
            model="inkoziev/rugpt_chitchat",
            device_map="auto"
        )

    def extract_inn_and_name(self, text):
        """Жесткий промпт с принудительным форматом"""

        prompt = f"""
        ### ИНСТРУКЦИЯ:
        Ты - система извлечения данных. Ты должна извлечь ИНН и ФИО из текста и вернуть ТОЛЬКО JSON.

        ### ТЕКСТ ДЛЯ АНАЛИЗА:
        "{text}"

        ### ФОРМАТ ОТВЕТА (ОБЯЗАТЕЛЬНО):
        ```json
        {{
          "ИНН": "найденный инн или null",
          "ФИО": "найденное фио или null" 
        }}
        ```

        ### ПРАВИЛА:
        1. ИНН - только цифры (10 или 12 символов)
        2. ФИО - фамилия, имя, отчество (русские буквы)
        3. Если данных нет - верни null

        ### НАЧИНАЙ ОТВЕТ С '```json' И ЗАКОНЧИ '```'

        ОТВЕТ:
        ```json
        """

        try:
            response = self.generator(
                prompt,
                max_new_tokens=200,
                temperature=0.1,  # Минимальная случайность
                do_sample=False,  # Без семплинга
                num_return_sequences=1,
                repetition_penalty=1.5,  # Штраф за повторения
                pad_token_id=self.generator.tokenizer.eos_token_id
            )[0]['generated_text']

            print("📨 СЫРОЙ ОТВЕТ МОДЕЛИ:")
            print(response)
            print("-" * 50)

            return self._bruteforce_parse(response, text)

        except Exception as e:
            return {"error": str(e), "raw": text}

    def _bruteforce_parse(self, response, original_text):
        """Агрессивный парсинг ответа модели"""

        # 1. Ищем JSON между ```json и ```
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # 2. Ищем любой JSON
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        # 3. Парсим вручную ключевые слова
        inn = self._extract_by_keywords(response, ["ИНН", "inn", "Инн"])
        fio = self._extract_by_keywords(response, ["ФИО", "фио", "Фамилия", "ФИО:"])

        return {
            "ИНН": inn if inn else None,
            "ФИО": fio if fio else None,
            "warning": "Парсинг через ключевые слова",
            "raw_response": response[:200] + "..." if len(response) > 200 else response
        }

    def _extract_by_keywords(self, text, keywords):
        """Извлекает значение после ключевых слов"""
        for keyword in keywords:
            pattern = f"{keyword}[\\s:]*([^\\n,\\.]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and value != "null":
                    return value
        return None


def test_hard_prompts():
    """Тестируем разные жесткие промпты"""

    extractor = ForceModelExtractor()

    test_cases = [
        "Аккр n 123, Инн 4353229845, Иванов Иван Иванович",
        "Клиент: Петров Алексей Сергеевич, ИНН 123456789012",
        "ФИО: Сидорова Мария, инн 9876543210",
        "Просто текст без данных"
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n{'🔴' * 20} ТЕСТ {i} {'🔴' * 20}")
        print(f"ВХОД: {text}")

        result = extractor.extract_inn_and_name(text)

        print(f"📊 РЕЗУЛЬТАТ:")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_hard_prompts()