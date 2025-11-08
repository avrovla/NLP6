import re
import json
from transformers import pipeline
from typing import Dict, Any


class SmartDataExtractor:
    def __init__(self):
        print("Инициализация умного экстрактора...")
        # Используем модель только для сложных случаев
        self.ai_helper = pipeline(
            "text-generation",
            model="inkoziev/rugpt_chitchat",
            device_map="auto"
        )

    def extract_inn_and_name(self, text: str) -> Dict[str, Any]:
        """Основной метод извлечения данных"""
        print(f"🔍 Анализируем текст: {text}")

        # 1. Rule-based извлечение (надежно)
        rule_based_result = self._rule_based_extraction(text)

        # 2. Если rule-based не нашел все данные, используем AI
        if not rule_based_result["ИНН"] or not rule_based_result["ФИО"]:
            print("⚠️  Rule-based не нашел все данные, подключаем AI...")
            ai_result = self._ai_assisted_extraction(text, rule_based_result)
            return ai_result

        print("✅ Данные найдены rule-based методом")
        return rule_based_result

    def _rule_based_extraction(self, text: str) -> Dict[str, Any]:
        """Извлечение данных по правилам"""
        result = {
            "ИНН": None,
            "ФИО": None,
            "метод": "rule-based",
            "исходный_текст": text
        }

        # Ищем ИНН (только цифры, длина 10 или 12)
        inn_matches = re.findall(r'\b\d{10,12}\b', text)
        if inn_matches:
            for inn in inn_matches:
                if len(inn) in [10, 12]:
                    result["ИНН"] = inn
                    break

        # Ищем ФИО (3 слова с заглавными буквами)
        fio_patterns = [
            r'\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)\b',  # Ф И О
            r'\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)\b',  # Ф И
        ]

        for pattern in fio_patterns:
            match = re.search(pattern, text)
            if match:
                result["ФИО"] = " ".join(match.groups())
                break

        return result

    def _ai_assisted_extraction(self, text: str, rule_result: Dict) -> Dict[str, Any]:
        """AI-помощник для сложных случаев"""
        prompt = f"""
        ТЕКСТ: "{text}"

        Уже найдено rule-based методом:
        - ИНН: {rule_result['ИНН'] or 'не найден'}
        - ФИО: {rule_result['ФИО'] or 'не найдено'}

        Помоги найти недостающие данные. Ответь ТОЛЬКО в формате:
        ИНН: <найденный_инн_или_пусто>
        ФИО: <найденное_фио_или_пусто>
        """

        try:
            response = self.ai_helper(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1
            )[0]['generated_text']

            # Парсим AI ответ
            ai_inn = self._extract_ai_value(response, "ИНН")
            ai_fio = self._extract_ai_value(response, "ФИО")

            # Объединяем с rule-based результатом
            final_result = rule_result.copy()
            final_result["метод"] = "hybrid"

            if not final_result["ИНН"] and ai_inn:
                final_result["ИНН"] = ai_inn
                final_result["AI_помощь_ИНН"] = True

            if not final_result["ФИО"] and ai_fio:
                final_result["ФИО"] = ai_fio
                final_result["AI_помощь_ФИО"] = True

            return final_result

        except Exception as e:
            print(f"❌ Ошибка AI: {e}")
            return rule_result

    def _extract_ai_value(self, response: str, field: str) -> str:
        """Извлекает значение из AI ответа"""
        pattern = f"{field}:\s*(.+)"
        match = re.search(pattern, response)
        if match:
            value = match.group(1).strip()
            # Очищаем значение
            if value.lower() in ['не найден', 'пусто', 'none', '']:
                return None
            return value
        return None


def main():
    extractor = SmartDataExtractor()

    print("🤖 Умный извлекатель данных")
    print("=" * 50)

    test_texts = [
        "Аккр n 123, Инн 4353229845, Иванов Иван Иванович",
        "Клиент: Петров Алексей Сергеевич, ИНН 123456789012",
        "ФИО: Сидорова Мария, инн 9876543210",
        "Просто какой-то текст без данных",
        "ИНН 1111111111 и имя John Doe",  # Английское имя
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n{'=' * 60}")
        print(f"ТЕСТ {i}: {text}")
        print('=' * 60)

        result = extractor.extract_inn_and_name(text)

        print("📊 РЕЗУЛЬТАТ:")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()