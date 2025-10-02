"""
Скрипт для сортировки изображений и их YOLO-меток по папкам на основе класса объекта.

Описание:
    Скрипт анализирует текстовые файлы меток в формате YOLO (расширение .txt),
    извлекает идентификатор класса из первой строки каждого файла (первое значение),
    и копирует как сам файл метки, так и соответствующее изображение (.jpg)
    в подпапку, названную по этому идентификатору класса.

    Например:
        Файл labels/train/abc.txt содержит строку: "3 0.5 0.5 0.2 0.3"
        → скрипт создаст папку sort/3/ и скопирует туда abc.txt и abc.jpg.

    Это полезно для:
      - визуального контроля качества разметки по классам,
      - ручной проверки или доаннотирования,
      - подготовки подвыборок по конкретным классам.

Требования:
    - Файлы меток должны быть в формате YOLO: <class_id> <x_center> <y_center> <width> <height>
    - Имена изображений и меток должны совпадать (например: image.jpg ↔ image.txt)
    - Поддерживаются только изображения в формате .jpg (можно адаптировать при необходимости)

Структура выходной директории:
    sort/
    ├── 0/
    │   ├── img1.jpg
    │   └── img1.txt
    ├── 1/
    │   ├── img2.jpg
    │   └── img2.txt
    └── ...

Примечания:
    - Если изображение не найдено, скрипт выведет предупреждение, но продолжит работу.
    - Пустые .txt-файлы пропускаются.
    - Идентификаторы классов сохраняются как строки (например, "0", "12"), что позволяет
      корректно обрабатывать случаи с ведущими нулями или нечисловыми метками (при наличии).
"""

import os
import shutil

# === НАСТРОЙКИ ===
# Укажите пути к исходным и целевым папкам
images_dir = 'Segmentation 1.50/images/train'   # Папка с изображениями (.jpg)
labels_dir = 'Segmentation 1.50/labels/train'   # Папка с YOLO-метками (.txt)
tergenate_dir = 'Segmentation 1.50/sort'        # Целевая папка для сортировки

# Проверяем, что обе исходные папки существуют
if not os.path.exists(images_dir):
    raise FileNotFoundError(f"Папка с изображениями '{images_dir}' не найдена.")
if not os.path.exists(labels_dir):
    raise FileNotFoundError(f"Папка с метками '{labels_dir}' не найдена.")

# Проходим по всем .txt файлам в папке меток
for filename in os.listdir(labels_dir):
    if filename.endswith('.txt'):
        txt_path = os.path.join(labels_dir, filename)
        
        try:
            # Читаем первую строку файла метки
            with open(txt_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    print(f"Файл {filename} пустой — пропускаем.")
                    continue
                
                # Извлекаем первый токен — предполагаем, что это идентификатор класса
                first_token = first_line.split()[0]
                class_id = first_token  # Сохраняем как строку для корректного имени папки
                
                # Создаём целевую папку для этого класса
                target_dir = os.path.join(tergenate_dir, class_id)
                os.makedirs(target_dir, exist_ok=True)
                
                # Копируем файл метки
                shutil.copy2(txt_path, os.path.join(target_dir, filename))
                
                # Формируем имя соответствующего изображения (.jpg)
                base_name = os.path.splitext(filename)[0]
                jpg_filename = base_name + '.jpg'
                jpg_path = os.path.join(images_dir, jpg_filename)
                
                # Копируем изображение, если оно существует
                if os.path.exists(jpg_path):
                    shutil.copy2(jpg_path, os.path.join(target_dir, jpg_filename))
                else:
                    print(f"Предупреждение: изображение {jpg_filename} не найдено для {filename}")
                    
        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")

print("✅ Обработка завершена. Файлы отсортированы по папкам классов.")