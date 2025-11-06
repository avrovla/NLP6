import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Tuple
import warnings

warnings.filterwarnings("ignore")


class RussianChatBot:
    def __init__(self, model_name: str = "inkoziev/rugpt_chitchat"):
        self.model_name = model_name
        self.history: List[Tuple[str, str]] = []

        print(f"Загрузка русскоязычной модели {model_name}...")

        # Используем пайплайн для простоты
        self.chat_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device_map="auto"
        )
        print("Русскоязычная модель загружена и готова к общению!")

    def chat(self, message: str) -> str:
        """Основной метод для общения с ботом"""
        try:
            # Формируем промпт в формате, который понимает модель
            prompt = self._build_prompt(message)

            # Генерируем ответ
            response = self.chat_pipeline(
                prompt,
                max_new_tokens=50,  # Ограничиваем длину ответа
                temperature=0.8,
                do_sample=True,
                num_return_sequences=1,
                repetition_penalty=1.1,
                truncation=True
            )[0]['generated_text']

            # Извлекаем только ответ бота
            bot_response = self._extract_bot_response(response, prompt)

            # Сохраняем в историю
            self.history.append((message, bot_response))

            # Ограничиваем размер истории
            if len(self.history) > 10:
                self.history.pop(0)

            return bot_response

        except Exception as e:
            return f"Извините, произошла ошибка: {str(e)}"

    def _build_prompt(self, new_message: str) -> str:
        """Строит промпт в правильном формате для модели"""
        prompt = ""

        # Добавляем историю диалога
        for user_msg, bot_msg in self.history[-3:]:  # Берем последние 3 обмена
            prompt += f"- {user_msg}\n- {bot_msg}\n"

        # Добавляем новое сообщение
        prompt += f"- {new_message}\n-"

        return prompt

    def _extract_bot_response(self, full_response: str, prompt: str) -> str:
        """Извлекает ответ бота из полного текста"""
        # Убираем промпт
        response = full_response.replace(prompt, "").strip()

        # Убираем все после следующего "-" (начало нового сообщения)
        if '\n-' in response:
            response = response.split('\n-')[0]

        # Очищаем от лишних пробелов
        response = ' '.join(response.split())

        return response if response else "Здравствуйте! Рад общению!"

    def clear_history(self):
        """Очистить историю диалога"""
        self.history.clear()
        print("История диалога очищена!")

    def get_history(self) -> List[Tuple[str, str]]:
        """Получить историю диалога"""
        return self.history.copy()


class LightweightChatBot:
    """Облегченная версия с той же моделью"""

    def __init__(self):
        print("Загрузка облегченного русскоязычного бота...")
        self.chat_pipeline = pipeline(
            "text-generation",
            model="inkoziev/rugpt_chitchat",
            device_map="auto"
        )
        self.history = []
        print("Бот загружен!")

    def chat(self, message: str) -> str:
        # Простой промпт без сложной истории
        prompt = f"- {message}\n-"

        response = self.chat_pipeline(
            prompt,
            max_new_tokens=40,
            temperature=0.9,
            do_sample=True,
            num_return_sequences=1
        )[0]['generated_text']

        bot_response = response.replace(prompt, "").strip()

        # Очищаем ответ
        if '\n' in bot_response:
            bot_response = bot_response.split('\n')[0]

        return bot_response if bot_response else "Привет!"

    def clear_history(self):
        self.history.clear()
        print("История очищена!")