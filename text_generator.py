from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Tuple, List

def load_tokenizer_and_model(model_name_or_path: str) -> Tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    """
    Загружает токенизатор и модель ruGPT-3.

    Параметры:
    model_name_or_path (str): Имя предобученной модели или путь к директории,
                              содержащей файлы модели и токенизатора.

    Возвращает:
    Tuple[GPT2Tokenizer, GPT2LMHeadModel]: Кортеж, содержащий загруженные объекты токенизатора и модели.

    Функция загружает указанный токенизатор и модель GPT с помощью библиотеки Transformers, 
    что позволяет использовать их для последующей генерации текста.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    return tokenizer, model

def generate_text(start_text: str, 
                  model: GPT2LMHeadModel, 
                  tokenizer: GPT2Tokenizer, 
                  max_length: int = 150, 
                  top_k: int = 5, 
                  top_p: float = 0.95, 
                  temperature: float = 1.0, 
                  num_return_sequences: int = 3, 
                  repetition_penalty: float = 1.5) -> List[str]:
    """
    Генерирует текст на основе начальной фразы с использованием модели GPT-2.

    Параметры:
    start_text (str): Начальный текст, который послужит отправной точкой для генерации.
    model (GPT2LMHeadModel): Загруженная модель GPT-2 для генерации текста.
    tokenizer (GPT2Tokenizer): Токенизатор, используемый для преобразования текста в токены.
    max_length (int): Максимальная длина сгенерированного текста в токенах, включая начальный текст. По умолчанию 150.
    top_k (int): Ограничивает выбор следующего слова из k самых вероятных вариантов. Используется в топ-k сэмплировании. По умолчанию 5.
    top_p (float): Используется в nucleus sampling или топ-p сэмплировании для выбора следующего слова из наиболее вероятных, суммированных до вероятности p. По умолчанию 0.95.
    temperature (float): Управляет вероятностным распределением следующих слов. Низкие значения делают текст более предсказуемым, высокие – более случайным. По умолчанию 1.0.
    num_return_sequences (int): Количество различных вариантов текста, которые следует сгенерировать. По умолчанию 3.
    repetition_penalty (float): Регулирует штраф за повторение, чтобы избежать генерации одинаковых фрагментов текста. По умолчанию 1.5.

    Возвращает:
    List[str]: Список сгенерированных текстов.
    """
    input_ids = tokenizer.encode(start_text, return_tensors="pt")
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]