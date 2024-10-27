import tkinter as tk
from tkinter import scrolledtext
from text_generator import generate_text

def setup_gui(tokenizer, model, config):
    """
    Настраивает и запускает интерфейс (GUI)
    """
    def generate_text_and_display():
        """
        Функция для генерации текста из интерфейса и отображения результата.
        """
        input_text = input_text_area.get("1.0", tk.END)
        
        # Параметры генерации, полученные из конфигурационного файла
        generation_params = config["generation_parameters"]
        
        # Вызываем функцию генерации текста с параметрами из config
        generated_texts = generate_text(input_text.strip(), model, tokenizer, generation_params)

        # Очищаем поле вывода
        output_text_area.delete("1.0", tk.END)

        # Отображаем каждый сгенерированный вариант текста
        for i, text in enumerate(generated_texts, 1):
            output_text_area.insert(tk.INSERT, f"Вариант {i}:\n{text}\n\n")
            print(f"Вариант {i}:\n{text}\n\n")

    # Настройка интерфейса GUI
    root = tk.Tk()
    root.title("Генерация текста (ruGPT-3)")

    # Поле с возможностью ввода текста (многострочное)
    tk.Label(root, text="Введите стартовую фразу:").pack(pady=5)
    input_text_area = scrolledtext.ScrolledText(root, width=80, height=10, wrap=tk.WORD)
    input_text_area.pack(pady=5)

    # Кнопка для запуска генерации текста
    generate_button = tk.Button(root, text="Сгенерировать текст", command=generate_text_and_display)
    generate_button.pack(pady=10)

    # Поле для отображения результата с возможностью прокрутки
    output_text_area = scrolledtext.ScrolledText(root, width=100, height=20, wrap=tk.WORD)
    output_text_area.pack(pady=10)

    # Запуск графического интерфейса
    root.mainloop()