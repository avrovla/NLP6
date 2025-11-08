from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import re


class OpenModelExtractor:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        print(f"🚀 Загрузка {model_name}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto"
            )

            # Создаем пайплайн
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )

            print(f"✅ {model_name} загружена!")

        except Exception as e:
            print(f"❌ Ошибка загрузки {model_name}: {e}")
            self._load_fallback_model()

    def _load_fallback_model(self):
        """Загрузка резервной модели"""
        print("🔄 Загрузка резервной модели...")
        self.generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-small",
            device_map="auto"
        )
        print("✅ Резервная модель загружена!")

    def extract_with_forced_json(self, text):
        """Принудительное извлечение с жестким промптом"""

        prompt = f"""
### SYSTEM: 
Ты - API. Ты принимаешь текст и возвращаешь JSON. 
Ты НЕ добавляешь никаких других слов кроме JSON.

### INPUT:
{text}

### OUTPUT FORMAT:
{"{"}
  "ИНН": "найденный инн или null",
  "ФИО": "найденное фио или null"
{"}"}

### RESPONSE (ONLY JSON):
"""

        try:
            response = self.generator(
                prompt,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                num_return_sequences=1,
                repetition_penalty=2.0,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]['generated_text']

            print("📨 СЫРОЙ ОТВЕТ:")
            print(response)
            print("-" * 50)

            return self._bruteforce_json_parse(response, text)

        except Exception as e:
            return {"error": str(e), "text": text}

    def _bruteforce_json_parse(self, response, original_text):
        """Агрессивный парсинг JSON"""

        # Метод 1: Ищем JSON между фигурными скобками
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "ИНН" in data or "ФИО" in data:
                    return self._validate_data(data, original_text)
            except:
                pass

        # Метод 2: Ищем ключевые слова и значения
        inn = self._extract_by_pattern(response, r'"?ИНН"?\s*[=:]\s*"([^"]*)"')
        fio = self._extract_by_pattern(response, r'"?ФИО"?\s*[=:]\s*"([^"]*)"')

        if inn or fio:
            return {
                "ИНН": self._clean_inn(inn),
                "ФИО": self._clean_fio(fio),
                "method": "pattern_matching",
                "original_text": original_text
            }

        # Метод 3: Ищем данные в исходном тексте
        return self._extract_from_original(original_text)

    def _extract_by_pattern(self, text, pattern):
        """Извлекает значение по паттерну"""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _clean_inn(self, inn):
        """Очищает ИНН"""
        if not inn:
            return None
        # Оставляем только цифры
        inn_clean = re.sub(r'\D', '', str(inn))
        return inn_clean if len(inn_clean) in [10, 12] else None

    def _clean_fio(self, fio):
        """Очищает ФИО"""
        if not fio:
            return None
        # Оставляем только русские буквы и пробелы
        fio_clean = re.sub(r'[^а-яА-ЯёЁ\s]', '', str(fio)).strip()
        return fio_clean if fio_clean else None

    def _extract_from_original(self, text):
        """Извлекает данные напрямую из исходного текста"""
        # Ищем ИНН
        inn_match = re.search(r'\b(\d{10,12})\b', text)
        inn = inn_match.group(1) if inn_match else None

        # Ищем ФИО (2-3 слова с заглавными)
        fio_match = re.search(r'([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})', text)
        fio = fio_match.group(1) if fio_match else None

        return {
            "ИНН": inn,
            "ФИО": fio,
            "method": "direct_extraction",
            "original_text": text
        }

    def _validate_data(self, data, original_text):
        """Валидирует извлеченные данные"""
        validated = data.copy()

        if "ИНН" in validated:
            validated["ИНН"] = self._clean_inn(validated["ИНН"])

        if "ФИО" in validated:
            validated["ФИО"] = self._clean_fio(validated["ФИО"])

        validated["method"] = "json_parsing"
        validated["original_text"] = original_text

        return validated


def test_open_models():
    """Тестируем разные открытые модели"""

    models_to_test = [
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-small",
        "gpt2",  # Самая простая, но надежная
    ]

    test_texts = [
        "Аккр n 123, Инн 4353229845, Иванов Иван Иванович",
        "Клиент: Петров Алексей Сергеевич, ИНН 123456789012",
        "ФИО: Сидорова Мария, инн 9876543210",
    ]

    for model_name in models_to_test:
        print(f"\n{'🎯' * 20} МОДЕЛЬ: {model_name} {'🎯' * 20}")

        try:
            extractor = OpenModelExtractor(model_name)

            for i, text in enumerate(test_texts, 1):
                print(f"\n📝 ТЕСТ {i}: {text}")
                result = extractor.extract_with_forced_json(text)
                print("📊 РЕЗУЛЬТАТ:")
                print(json.dumps(result, ensure_ascii=False, indent=2))

        except Exception as e:
            print(f"❌ Ошибка с {model_name}: {e}")
            continue


if __name__ == "__main__":
    test_open_models()