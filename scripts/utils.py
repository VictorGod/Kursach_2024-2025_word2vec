import os
import json
from gensim.models import Word2Vec
from PyPDF2 import PdfReader

def read_file(file_path):
    """Читает содержимое текстового файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(file_path, content):
    """Записывает содержимое в текстовый файл."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def load_json(file_path):
    """Загружает данные из JSON файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(file_path, data):
    """Сохраняет данные в JSON файл."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_text_from_pdf(pdf_path):
    """Извлечение текста из PDF-файла."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    """Очистка текста: приведение к нижнему регистру, удаление лишних пробелов."""
    text = text.lower()
    text = ' '.join(text.split())
    return text

def train_word2vec(data_file, model_path, vector_size=100, window=5, min_count=2, workers=4):
    """Обучение модели Word2Vec."""
    with open(data_file, 'r', encoding='utf-8') as f:
        sentences = [line.split() for line in f]

    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    model.save(model_path)
    print(f"Модель сохранена в {model_path}")

def load_word2vec_model(model_path):
    """Загрузка модели Word2Vec."""
    return Word2Vec.load(model_path)

def get_word_vector(model, word):
    """Получение вектора слова из модели."""
    if word in model.wv:
        return model.wv[word]
    return None

def get_average_vector(model, text):
    """Получение среднего вектора текста."""
    words = text.split()
    vectors = [get_word_vector(model, word) for word in words if get_word_vector(model, word) is not None]
    if not vectors:
        return None
    return sum(vectors) / len(vectors)

def ensure_directory_exists(directory):
    """Создает директорию, если она не существует."""
    os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    print("Utils module loaded.")
