from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


# 📁 Загрузка токена из файла
def load_token():
    try:
        with open("token.txt", "r", encoding="utf-8") as f:
            token = f.read().strip()
        if not token:
            raise ValueError("Токен пустой")
        return token
    except FileNotFoundError:
        print("❌ Файл token.txt не найден!")
        print("📝 Создайте файл token.txt и положите туда ваш Hugging Face токен")
        return None
    except Exception as e:
        print(f"❌ Ошибка загрузки токена: {e}")
        return None


# 🔐 Загрузка токена
TOKEN = load_token()
if not TOKEN:
    exit()

# 🎯 Загрузка модели
print("🚀 Загружаю Gemma...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    torch_dtype=torch.float16,
    device_map="auto",
    token=TOKEN
)
print("✅ Gemma загружена!")


# 💬 Простой чат
def chat_with_gemma(message):
    # Формируем промпт в формате Gemma
    prompt = f"""<start_of_turn>user
{message}<end_of_turn>
<start_of_turn>model
"""

    # Токенизируем
    inputs = tokenizer(prompt, return_tensors="pt")

    # Генерируем ответ
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7
        )

    # Декодируем ответ
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Извлекаем только ответ модели
    return response.split("<start_of_turn>model")[-1].strip()


# 🎪 Тестируем
messages = [
    "Привет! Как дела?",
    "Напиши короткое стихотворение о Python",
    "Объясни что такое искусственный интеллект простыми словами"
]

for message in messages:
    print(f"👤 Вы: {message}")
    response = chat_with_gemma(message)
    print(f"🤖 Gemma: {response}")
    print("-" * 50)