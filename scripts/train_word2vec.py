from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


def train_word2vec(data_file, model_path):
    """
    Обучение модели Word2Vec.
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        sentences = [simple_preprocess(line) for line in f]

    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    model.save(model_path)
    print(f"Модель сохранена в {model_path}")


if __name__ == "__main__":
    # Обучение модели для медицинских статей
    train_word2vec("data/medical_dataset.txt", "models/medical_word2vec.model")
    
    # Обучение модели для статей по ВКР
    train_word2vec("data/vkr_dataset.txt", "models/vkr_word2vec.model")
