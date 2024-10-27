import json

def load_config(config_path: str = 'config.json'):
    """
    Загружает конфигурационные параметры из файла config.json

    :param config_path: Путь к файлу конфигурации, по умолчанию 'config.json'
    :return: Словарь с конфигурационными данными
    """
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
    return config