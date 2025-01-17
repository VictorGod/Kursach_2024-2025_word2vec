import os
from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path):
    """
    Извлечение текста из PDF-файла.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def clean_text(text):
    """
    Предобработка текста:
    - Приведение к нижнему регистру
    - Удаление лишних пробелов
    """
    text = text.lower()
    text = ' '.join(text.split())
    return text


def process_pdfs(input_folder, output_file):
    """
    Обработка PDF-файлов: извлечение текста, очистка и сохранение в .txt файл.
    """
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.pdf'):
                pdf_path = os.path.join(input_folder, file_name)
                text = extract_text_from_pdf(pdf_path)
                clean_text_content = clean_text(text)
                f_out.write(clean_text_content + "\n")


if __name__ == "__main__":
    # Обработка медицинских статей
    process_pdfs("data/medical_articles", "data/medical_dataset.txt")
    
    # Обработка статей по ВКР
    process_pdfs("data/vkr_articles", "data/vkr_dataset.txt")
