import tkinter as tk
from tkinter import scrolledtext
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Tuple, List


def load_tokenizer_and_model(model_name_or_path: str) -> Tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    """
    Загружает токенизатор и модель ruGPT-3

    :param model_name_or_path: Путь или название предобученной модели GPT-3
    :return: Кортеж из токенизатора и модели
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
                  num_return_sequences: int = 3) -> List[str]:
    """
    Генерирует текст на основе стартовой фразы

    :param start_text: Исходный текст для генерации
    :param model: Загрузка NLP модели GPT-3
    :param tokenizer: Токенизатор для переводов текстов в тензоры
    :param max_length: Максимальная длина генерируемого текста (по умолчанию 150 символов)
    :param top_k: Параметр top-k для генерации (контролирует, сколько токенов будет выбрано для сэмплирования)
    :param top_p: Параметр top-p для генерации (контролирует вероятность удаления низкочастотных токенов)
    :param temperature: Температура генерации, контролирует разнообразие выбора
    :param num_return_sequences: Количество вариантов текста для генерации
    :return: Список сгенерированных текстов
    """
    input_ids = tokenizer.encode(start_text, return_tensors="pt")
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=1.5,
        do_sample=True,
        num_return_sequences=num_return_sequences
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


def generate_text_and_display():
    """
    Функция для генерации текста из интерфейса и отображения результата
    """
    input_text = input_text_area.get("1.0", tk.END)
    generated_texts = generate_text(input_text.strip(), model, tokenizer)

    # Очищаем поле вывода
    output_text_area.delete("1.0", tk.END)

    # Отображаем каждый сгенерированный вариант текста
    for i, text in enumerate(generated_texts, 1):
        output_text_area.insert(tk.INSERT, f"Вариант {i}:\n{text}\n\n")
        print(f"Вариант {i}:\n{text}\n\n")


# Настройка интерфейса GUI
root = tk.Tk()
root.title("Генерация текста (ruGPT-3)")

# Поле с возможностью ввода и прокрутки для стартового текста (многострочное)
tk.Label(root, text="Введите стартовую фразу:").pack(pady=5)
input_text_area = scrolledtext.ScrolledText(root, width=80, height=10, wrap=tk.WORD)
input_text_area.pack(pady=5)

# Кнопка для запуска генерации текста
generate_button = tk.Button(root, text="Сгенерировать текст", command=generate_text_and_display)
generate_button.pack(pady=10)

# Поле для отображения сгенерированного текста с возможностью прокрутки
output_text_area = scrolledtext.ScrolledText(root, width=100, height=20, wrap=tk.WORD)
output_text_area.pack(pady=10)

# Загрузка модели и токенизатора ruGPT-3 из библиотеки transformers
model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer, model = load_tokenizer_and_model(model_name_or_path)

# Запуск GUI приложения
root.mainloop()