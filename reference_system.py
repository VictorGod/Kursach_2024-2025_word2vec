import json
import numpy as np
import re
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

REFERENCE_FILE = "reference_data.json"

# Хэш-таблица для быстрого поиска ключевых терминов
HASHMAP_DB = {
    "диабет": "Диабет - хроническое заболевание, связанное с высоким уровнем сахара в крови.",
    "автоэнкодеры": "Автоэнкодеры - это нейронные сети, используемые для снижения размерности данных и генерации новых представлений."
}

def load_reference_data(file_path=REFERENCE_FILE):
    """Загрузка категорий и векторов из JSON."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Ошибка: reference_data.json не найден.")
        return None
    except json.JSONDecodeError:
        print("Ошибка: Ошибка в формате JSON.")
        return None

def preprocess_text(text):
    """Очистка и токенизация текста."""
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)
    return words

def query_to_vector(query, model_path):
    """Преобразование запроса в средний вектор с использованием модели Word2Vec."""
    try:
        model = Word2Vec.load(model_path)
    except FileNotFoundError:
        print(f"Ошибка: модель {model_path} не найдена.")
        return None
    words = preprocess_text(query)
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        print(f"Все слова в запросе '{query}' отсутствуют в модели!")
        return None
    return np.mean(vectors, axis=0)

def find_closest_category(query, reference_data):
    """Находит ближайшую категорию по косинусному сходству."""
    best_match = None
    highest_similarity = -1
    query_vector = None
    for category in reference_data:
        model_path = f"models/{category['category_name'].strip()}_word2vec.model"
        query_vector = query_to_vector(query, model_path)
        if query_vector is not None:
            break
    if query_vector is None:
        return None, None
    for category in reference_data:
        category_vector = np.array(category["avg_vector"])
        similarity = 1 - cosine(query_vector, category_vector)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = category
    return best_match, highest_similarity if best_match else None, None

if __name__ == "__main__":
    reference_data = load_reference_data()
    if reference_data:
        while True:
            user_query = input("\nВведите ваш запрос (или 'выход' для завершения): ")
            if user_query.lower() in ["выход", "exit"]:
                print("Выход из системы.")
                break
            
            closest_category, similarity = find_closest_category(user_query, reference_data)
            
            # Проверка хэш-таблицы, если не найдена категория
            if closest_category is None:
                print("Не удалось найти категорию, пробуем поиск в HashMap...")
                for key in HASHMAP_DB.keys():
                    if key in user_query.lower():
                        print(f"\n Найдено в HashMap: {key}")
                        print(f"Ответ: {HASHMAP_DB[key]}")
                        break
                else:
                    print("Нет информации по данному запросу.")
                continue
            
            response = closest_category["description"]
            print(f"\n Ближайшая категория: {closest_category['category_name']} (Сходство: {similarity:.2f})")
            print(f"Ответ: {response}")
            print("Ключевые термины:")
            for term in closest_category["key_terms"]:
                print(f" - {term['term']}: {term['description']} (сходство: {term['similarity']:.2f})")
