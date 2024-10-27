from gui import setup_gui
from text_generator import load_tokenizer_and_model
from config import load_config

if __name__ == '__main__':
    # Загрузка конфигурации
    config = load_config()

    # Получаем путь к модели из конфигурационного файла
    model_name_or_path = config["model_name_or_path"]

    # Загрузка модели и токенизатора
    tokenizer, model = load_tokenizer_and_model(model_name_or_path)

    # Запуск GUI приложения с переданными конфигурациями
    setup_gui(tokenizer, model, config)
