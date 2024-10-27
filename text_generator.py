from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Tuple, List

def load_tokenizer_and_model(model_name_or_path: str) -> Tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    """
    Загружает токенизатор и модель ruGPT-3
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