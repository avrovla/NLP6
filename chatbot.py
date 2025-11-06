import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Tuple
import warnings

warnings.filterwarnings("ignore")


class FastChatBot:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.history: List[Tuple[str, str]] = []

        print(f"Загрузка модели {model_name}...")

        # Используем 4-битное квантование для скорости и экономии памяти
        self.chat_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,  # Ускоряет работу в 2-3 раза
            low_cpu_mem_usage=True
        )

        print("Модель загружена и готова к работе!")

    def chat(self, message: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Основной метод для общения с ботом"""

        # Формируем контекст из истории
        context = ""
        for user_msg, bot_msg in self.history[-4:]:  # Берем последние 4 обмена
            context += f"Пользователь: {user_msg}\nБот: {bot_msg}\n"

        prompt = f"{context}Пользователь: {message}\nБот:"

        # Генерируем ответ
        with torch.no_grad():  # Отключаем градиенты для ускорения
            response = self.chat_pipeline(
                prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.chat_pipeline.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )[0]['generated_text']

        # Извлекаем только новый ответ
        bot_response = response.replace(prompt, "").strip()

        # Сохраняем в историю
        self.history.append((message, bot_response))

        return bot_response

    def clear_history(self):
        """Очистить историю диалога"""
        self.history.clear()
        print("История диалога очищена!")

    def get_history(self) -> List[Tuple[str, str]]:
        """Получить историю диалога"""
        return self.history.copy()


class LightweightChatBot:
    """Сверхлегкая версия для слабых компьютеров"""

    def __init__(self):
        print("Загрузка легкой модели...")

        # Самая быстрая модель для диалогов
        self.chat_pipeline = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-small",  # Очень легкая
            device_map="auto"
        )
        print("Легкая модель загружена!")

    def chat(self, message: str) -> str:
        prompt = f"Пользователь: {message}\nБот:"

        response = self.chat_pipeline(
            prompt,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
            pad_token_id=self.chat_pipeline.tokenizer.eos_token_id
        )[0]['generated_text']

        return response.replace(prompt, "").strip()