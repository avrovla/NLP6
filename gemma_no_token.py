from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class GemmaChat:
    def __init__(self):
        print("🚀 Загружаю Gemma...")
        try:
            # Пробуем без токена
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
            self.model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-2b-it",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✅ Gemma загружена без токена!")
            self.ready = True

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("📝 Модель не найдена в кеше. Нужен токен для первой загрузки.")
            self.ready = False

    def chat(self, message):
        if not self.ready:
            return "❌ Модель не загружена"

        prompt = f"""<start_of_turn>user
{message}<end_of_turn>
<start_of_turn>model
"""

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<start_of_turn>model")[-1].strip()


# 🎪 Тест
if __name__ == "__main__":
    bot = GemmaChat()

    if bot.ready:
        print("\n🤖 Gemma готова к общению!")

        test_messages = [
            "Привет! Как дела?",
            "Расскажи о себе",
            "Что ты умеешь?"
        ]

        for msg in test_messages:
            print(f"\n👤 Вы: {msg}")
            response = bot.chat(msg)
            print(f"🤖 Gemma: {response}")