from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import re


class GemmaExtractor:
    def __init__(self):
        print("🚀 Загрузка Gemma-2-2b-it...")

        model_name = "google/gemma-2-2b-it"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Создаем пайплайн для удобства
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

        print("✅ Gemma загружена!")

    def extract_inn_and_name(self, text):
        """Извлекает данные с помощью Gemma"""

        prompt = f"""<start_of_turn>user
Извлеки ИНН и ФИО из текста и верни ТОЛЬКО JSON. Не добавляй никакого текста.

Текст: "{text}"

Формат ответа:
{{
  "ИНН": "найденный инн или null",
  "ФИО": "найденное фио или null"
}}<end_of_turn>
<start_of_turn>model
"""

        try:
            response = self.generator(
                prompt,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]['generated_text']

            print("📨 СЫРОЙ ОТВЕТ GEMMA:")
            print(response)
            print("-" * 50)

            # Извлекаем только ответ модели
            model_response = response.split("<start_of_turn>model")[-1].strip()

            return self._parse_gemma_response(model_response, text)

        except Exception as e:
            return {"error": str(e), "text": text}

    def extract_with_chat_template(self, text):
        """Используем чат-темплейт Gemma"""

        messages = [
            {"role": "user", "content": f"""Извлеки ИНН и ФИО из этого текста и верни ТОЛЬКО JSON:

Текст: {text}

Формат:
{{
  "ИНН": "найденный инн или null", 
  "ФИО": "найденное фио или null"
}}"""}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлекаем ответ модели
        model_part = response.split("<start_of_turn>model")[-1].strip()

        print("💬 CHAT TEMPLATE RESPONSE:")
        print(model_part)

        return self._parse_gemma_response(model_part, text)

    def _parse_gemma_response(self, response, original_text):
        """Парсит ответ Gemma"""
        try:
            # Ищем JSON
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())

                # Валидация данных
                if "ИНН" in data and data["ИНН"]:
                    # Оставляем только цифры
                    inn_clean = re.sub(r'\D', '', str(data["ИНН"]))
                    if len(inn_clean) in [10, 12]:
                        data["ИНН"] = inn_clean
                    else:
                        data["ИНН"] = None

                if "ФИО" in data and data["ФИО"]:
                    # Очищаем ФИО
                    fio_clean = re.sub(r'[^а-яА-ЯёЁ\s]', '', str(data["ФИО"])).strip()
                    data["ФИО"] = fio_clean if fio_clean else None

                return data
            else:
                return {
                    "error": "JSON не найден в ответе",
                    "raw_response": response,
                    "original_text": original_text
                }

        except json.JSONDecodeError as e:
            return {
                "error": f"Ошибка парсинга JSON: {e}",
                "raw_response": response,
                "original_text": original_text
            }


def test_gemma():
    """Тестируем Gemma"""

    extractor = GemmaExtractor()

    test_cases = [
        "Аккр n 123, Инн 4353229845, Иванов Иван Иванович",
        "Клиент: Петров Алексей Сергеевич, ИНН 123456789012",
        "ФИО: Сидорова Мария, инн 9876543210",
        "Просто текст без структурированных данных",
        "ИНН 1111111111 для John Doe"
    ]

    print("🧪 ТЕСТИРУЕМ GEMMA-2-2b-it")
    print("=" * 60)

    for i, text in enumerate(test_cases, 1):
        print(f"\n🎯 ТЕСТ {i}: {text}")

        # Пробуем оба метода
        print("\n1. 📝 Прямой промпт:")
        result1 = extractor.extract_inn_and_name(text)
        print(json.dumps(result1, ensure_ascii=False, indent=2))

        print("\n2. 💬 Чат-темплейт:")
        result2 = extractor.extract_with_chat_template(text)
        print(json.dumps(result2, ensure_ascii=False, indent=2))

        print("\n" + "─" * 50)


if __name__ == "__main__":
    test_gemma()