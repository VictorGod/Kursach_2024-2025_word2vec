import os
import numpy as np
from gensim.models import Word2Vec
import PyPDF2
import re

def extract_text_from_pdf(pdf_path):
    """
    Извлекает текст из PDF файла.
    """
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            return text
    except Exception as e:
        print(f"Ошибка при обработке файла {pdf_path}: {e}")
        return ""

def normalize_vector(vector):
    """
    Нормализует вектор (приводит к единичной длине).
    """
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def categorize_text(text, model):
    """
    Категоризация текста с использованием Word2Vec.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    vectors = [model.wv[word] for word in words if word in model.wv]

    if not vectors:
        return None, words  # Вернуть пустой вектор, если нет известных слов

    avg_vector = np.mean(vectors, axis=0)
    avg_vector = normalize_vector(avg_vector)
    missing_words = [word for word in words if word not in model.wv]
    return avg_vector, missing_words

def process_dataset(dataset_path, model_path):
    """
    Обрабатывает датасет и возвращает векторы для каждого текста.
    """
    model = Word2Vec.load(model_path)
    file_vectors = {}

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                text = extract_text_from_pdf(file_path)
                avg_vector, missing_words = categorize_text(text, model)

                if avg_vector is not None:
                    file_vectors[file] = avg_vector
                else:
                    print(f"В файле {file} отсутствуют слова, представленные в модели.")

    return file_vectors

def categorize_and_sort(vectors):
    """
    Сортирует тексты на две категории по значениям нормализованных векторов.
    """
    normalized_vectors = {file: normalize_vector(vec) for file, vec in vectors.items()}
    sorted_vectors = sorted(normalized_vectors.items(), key=lambda x: np.linalg.norm(x[1]))
    midpoint = len(sorted_vectors) // 2

    category_1 = sorted_vectors[:midpoint]
    category_2 = sorted_vectors[midpoint:]

    avg_category_1 = normalize_vector(np.mean([vec for _, vec in category_1], axis=0))
    avg_category_2 = normalize_vector(np.mean([vec for _, vec in category_2], axis=0))

    return category_1, category_2, avg_category_1, avg_category_2

def get_closest_words(model, vector, topn=5):
    """
    Возвращает ближайшие слова для заданного вектора.
    """
    return model.wv.similar_by_vector(vector, topn=topn)

if __name__ == "__main__":
    medical_dataset_path = "data/medical_articles"
    vkr_dataset_path = "data/vkr_articles"
    medical_model_path = "models/medical_word2vec.model"
    vkr_model_path = "models/vkr_word2vec.model"

    # Обработка медицинского датасета
    print("Обработка медицинских статей...")
    medical_vectors = process_dataset(medical_dataset_path, medical_model_path)
    
    # Обработка ВКР датасета
    print("Обработка статей ВКР...")
    vkr_vectors = process_dataset(vkr_dataset_path, vkr_model_path)

    # Рубрицирование медицинских статей
    print("Рубрицирование медицинских статей...")
    med_cat1, med_cat2, med_avg1, med_avg2 = categorize_and_sort(medical_vectors)

    # Рубрицирование статей ВКР
    print("Рубрицирование статей ВКР...")
    vkr_cat1, vkr_cat2, vkr_avg1, vkr_avg2 = categorize_and_sort(vkr_vectors)

    # Вывод результатов
    print("Медицинская категория 1 (средний вектор):", med_avg1)
    print("Медицинская категория 2 (средний вектор):", med_avg2)

    print("ВКР категория 1 (средний вектор):", vkr_avg1)
    print("ВКР категория 2 (средний вектор):", vkr_avg2)

    # Получение ключевых слов
    medical_model = Word2Vec.load(medical_model_path)
    vkr_model = Word2Vec.load(vkr_model_path)

    print("Ключевые слова для медицинской категории 1:", get_closest_words(medical_model, med_avg1))
    print("Ключевые слова для медицинской категории 2:", get_closest_words(medical_model, med_avg2))

    print("Ключевые слова для категории ВКР 1:", get_closest_words(vkr_model, vkr_avg1))
    print("Ключевые слова для категории ВКР 2:", get_closest_words(vkr_model, vkr_avg2))
