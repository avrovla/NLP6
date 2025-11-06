import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


class CopiedGemmaChat:
    def __init__(self):
        # Путь к скопированному каталогу
        self.model_path = Path("models") / "models--google--gemma-2-2b-it"

        print("🚀 Загрузка Gemma из скопированного каталога...")
        print(f"📁 Путь: {self.model_path}")

        if not self.model_path.exists():
            print("❌ Каталог модели не найден!")
            self.ready = False
            return

        # Проверим что внутри есть snapshots
        snapshots_path = self.model_path / "snapshots"
        if not snapshots_path.exists():
            print("❌ Папка snapshots не найдена!")
            self.ready = False
            return

        # Найдем папку с хешем (первая в snapshots)
        snapshot_dirs = list(snapshots_path.iterdir())
        if not snapshot_dirs:
            print("❌ Нет snapshot'ов в папке")
            self.ready = False
            return

        # Путь к реальным файлам модели
        self.actual_model_path = snapshot_dirs[0]
        print(f"📁 Файлы модели в: {self.actual_model_path}")

        try:
            # Загружаем из папки с файлами модели
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.actual_model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.actual_model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True  # Только локальные файлы
            )

            print("✅ Gemma загружена из скопированного каталога!")
            self.ready = True

        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            self.ready = False

    def chat(self, message: str) -> str:
        if not self.ready:
            return "❌ Модель не загружена."

        try:
            # Простой промпт
            prompt = f"Вопрос: {message}\nОтвет:"

            inputs = self.tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            bot_response = full_response.replace(prompt, "").strip()

            # Очистка ответа
            if '\n' in bot_response:
                bot_response = bot_response.split('\n')[0]

            return bot_response if bot_response else "Не могу ответить"

        except Exception as e:
            return f"❌ Ошибка: {str(e)}"


def main():
    bot = CopiedGemmaChat()

    if not bot.ready:
        return

    print("\n" + "=" * 50)
    print("🤖 Gemma из скопированного каталога")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("👋 До свидания!")
                break

            print("🤖 Gemma: ", end="", flush=True)
            response = bot.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break


if __name__ == "__main__":
    main()