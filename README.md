### README: Курсовая работа – Автоматическое рубрицирование текстов на основе Word2Vec  

####  Описание проекта  
Данный проект представляет собой курсовую работу по дисциплине *«Методы работы с большими данными»*, посвященную автоматическому рубрицированию текстов с использованием модели **Word2Vec**. В ходе работы была разработана система, способная классифицировать текстовые данные, структурировать информацию и интегрировать полученные результаты в справочные и диалоговые системы.  

####  Основные задачи и их выполнение  
✅ **Задание 1**: **Сбор текстовых данных**  
- Определена тематика курсовой работы на основе ВКР.  
- Собраны статьи из научных источников и распределены по двум категориям: медицинская тематика и ВКР.  

✅ **Задание 2**: **Создание датасета**  
- Статьи конвертированы в текстовый формат с очисткой от лишних элементов (метаданные, ссылки и т. д.).  
- Подготовлены два датасета: для медицинской тематики и ВКР.  

✅ **Задание 3**: **Обучение модели Word2Vec**  
- Обучена векторная модель (vector_size=100, window=5, min_count=2).  
- Проверена корректность работы модели с помощью тестовых слов.  

✅ **Задание 4**: **Автоматическое рубрицирование текстов**  
- Документы разбиты на массив слов, токенизированы и векторизованы.  
- Вычислены средние векторы текстов, на их основе произведено рубрицирование.  
- Найдены ключевые слова и определены категории.  

✅ **Задание 5**: **Сбор справочных данных**  
- Составлены справочные описания рубрик с ключевыми терминами.  
- Создана база справочных данных в формате JSON.  

✅ **Задание 6**: **Создание автоматической справочной системы**  
- Разработан алгоритм обработки запросов: сопоставление вектора запроса с базой данных.  
- Запрос пользователя анализируется и возвращается наиболее релевантная информация.  

✅ **Задание 7**: **Разработка диалоговой системы**  
- Создан файл `intents.json` с возможными вопросами и ответами чат-бота.  
- Разработан алгоритм ответа на пользовательские запросы с помощью fuzzy matching.  
- Чат-бот адаптирован для поиска информации в базе данных.  

####  Техническая реализация  
- **Обработка текстов**: конвертация из PDF, токенизация, удаление стоп-слов.  
- **Векторизация**: обучение модели Word2Vec.  
- **Классификация**: сравнение средних векторов документов.  
- **Справочная система**: база данных JSON, содержащая рубрики и ключевые слова.  
- **Чат-бот**: обработка пользовательских запросов и ответы на вопросы.  

####  Файлы проекта  
- `data/` – подготовленные текстовые данные.  
- `models/` – обученная модель Word2Vec.  
- `reference_data.json` – база справочных данных.  
- `intents.json` – конфигурация чат-бота.  
- `dialog_system.py` – реализация диалоговой системы.  

####  Используемые технологии  
- **Python** (Gensim, NumPy, PyPDF2, FuzzyWuzzy)  
- **JSON** для хранения данных  
- **NLP и машинное обучение** (Word2Vec)  

####  Итог  
Все задания курсовой работы успешно выполнены. Разработанная система позволяет автоматически классифицировать текстовые данные, структурировать информацию и интегрировать её в справочные и диалоговые системы. В дальнейшем проект может быть расширен за счет более сложных моделей и улучшения обработки запросов.
