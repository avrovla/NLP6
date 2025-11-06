from chatbot import RussianChatBot, LightweightChatBot


def main():
    print("🚀 Русскоязычный чат-бот")
    print("=" * 40)

    # Выбор модели
    print("Выберите режим:")
    print("1. Полноценный чат (с историей диалога)")
    print("2. Быстрый чат (без истории)")

    choice = input("Ваш выбор (1/2): ").strip()

    if choice == "1":
        bot = RussianChatBot()
    else:
        bot = LightweightChatBot()

    print("\n🤖 Бот готов к общению на русском!")
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
                print("История очищена!")
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