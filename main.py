from chatbot import FastChatBot, LightweightChatBot
import os


def main():
    print("🚀 Локальный чат-бот")
    print("=" * 40)

    # Выбор модели
    print("Выберите модель:")
    print("1. Быстрая (DialoGPT-medium) - рекомендуется")
    print("2. Сверхлегкая (DialoGPT-small) - для слабых ПК")
    print("3. Умная (Phi-3 mini) - требует скачивания ~4GB")

    choice = input("Ваш выбор (1/2/3): ").strip()

    if choice == "1":
        bot = FastChatBot("microsoft/DialoGPT-medium")
    elif choice == "2":
        bot = LightweightChatBot()
    elif choice == "3":
        bot = FastChatBot("microsoft/Phi-3-mini-4k-instruct")
    else:
        print("Используем модель по умолчанию...")
        bot = FastChatBot()

    print("\n🤖 Бот готов к общению!")
    print("Команды: 'очистить' - очистить историю, 'выход' - завершить")
    print("-" * 50)

    # Основной цикл чата
    while True:
        try:
            user_input = input("\nВы: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("До свидания!")
                break
            elif user_input.lower() in ['очистить', 'clear']:
                bot.clear_history()
                continue

            # Получаем ответ от бота
            print("Бот: ", end="", flush=True)
            response = bot.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\nПрограмма прервана. До свидания!")
            break
        except Exception as e:
            print(f"\n⚠️ Ошибка: {e}")
            print("Попробуйте еще раз...")


if __name__ == "__main__":
    main()