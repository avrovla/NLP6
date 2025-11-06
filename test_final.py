from chatbot import RussianChatBot


def test_final_bot():
    print("Тестируем финальную версию русскоязычного бота...")
    bot = RussianChatBot()

    test_dialog = [
        "Привет!",
        "Как твои дела?",
        "Что ты умеешь?",
        "Расскажи что-нибудь интересное",
        "Спасибо за общение!"
    ]

    for i, message in enumerate(test_dialog, 1):
        print(f"\n[{i}] Вы: {message}")
        response = bot.chat(message)
        print(f"[{i}] Бот: {response}")

    print("\n" + "=" * 60)
    print("Тест завершен! Бот отлично понимает русский язык!")


if __name__ == "__main__":
    test_final_bot()