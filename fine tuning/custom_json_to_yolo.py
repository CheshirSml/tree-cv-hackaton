"""
Конвертер пользовательских JSON-аннотаций в формат YOLO.

Описание:
    Скрипт преобразует аннотации изображений, хранящиеся в виде JSON-файлов
    с ключами "bbox" (в пикселях) и "type" (имя класса), в стандартный формат
    YOLO, пригодный для обучения детекторов объектов.

    Для каждого изображения:
      - Ищется соответствующий JSON-файл с тем же именем (но расширением .json).
      - Проверяется существование изображения и корректность его размеров.
      - Bounding box конвертируется из формата [x1, y1, x2, y2] →
        [x_center, y_center, width, height] с нормализацией относительно ширины и высоты изображения.
      - Классы автоматически нумеруются по алфавиту.

    Результат:
      - Структура папок: 
          yolo_dataset/
            ├── images/train/
            ├── images/val/
            ├── labels/train/
            ├── labels/val/
            └── dataset.yaml
      - Файл dataset.yaml содержит метаданные датасета (число классов, их имена и пути).

Формат входного JSON-файла (пример):
    {
        "bbox": [100, 150, 300, 400],
        "type": "sanitarka"
    }

Требования:
    - Изображения и JSON-файлы должны иметь одинаковые имена (например: image.jpg ↔ image.json).
    - Поддерживаются только изображения, читаемые библиотекой PIL (JPEG, PNG и др.).

Зависимости:
    - Pillow (PIL): pip install Pillow

Использование:
    1. Укажите пути к папкам с изображениями, JSON-аннотациями и выходной директорией.
    2. Настройте split_ratio (по умолчанию 0.8 — 80% train, 20% val).
    3. Запустите скрипт.
"""

import json
from pathlib import Path
import shutil
from PIL import Image
import random


def custom_json_to_yolo(images_dir, jsons_dir, output_dir, split_ratio=0.8):
    """
    Конвертирует пользовательские JSON-аннотации в формат YOLO.

    Аргументы:
        images_dir (str или Path): Путь к папке с изображениями.
        jsons_dir (str или Path): Путь к папке с JSON-аннотациями.
        output_dir (str или Path): Путь к выходной директории для YOLO-датасета.
        split_ratio (float): Доля данных для обучающей выборки (остальное — валидация).
    """
    images_dir = Path(images_dir)
    jsons_dir = Path(jsons_dir)
    output_dir = Path(output_dir)

    # Найти все JSON-файлы
    json_files = list(jsons_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"❌ Не найдено JSON-файлов в {jsons_dir}")

    print(f"Найдено JSON-файлов: {len(json_files)}")

    # Сбор валидных данных и классов
    valid_data = []
    all_types = set()

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Ошибка чтения JSON {json_path}: {e}")
            continue

        if not isinstance(data, dict):
            print(f"⚠️ Пропущен {json_path}: JSON не является объектом")
            continue

        if "bbox" not in data or "type" not in data:
            print(f"⚠️ Пропущен {json_path}: отсутствует 'bbox' или 'type'")
            continue

        # Извлекаем имя изображения: убираем только '.json' с конца
        if json_path.name.endswith('.json'):
            img_name = json_path.name[:-5]  # например: '_9FMvuan7S8.jpg'
        else:
            print(f"⚠️ Пропущен {json_path}: не заканчивается на .json")
            continue

        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"⚠️ Изображение не найдено: {img_path}")
            continue

        # Проверяем размеры изображения
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            if w == 0 or h == 0:
                raise ValueError("Нулевые размеры")
        except Exception as e:
            print(f"❌ Ошибка открытия изображения {img_path}: {e}")
            continue

        # Сохраняем данные
        all_types.add(data["type"])
        valid_data.append({
            "json_path": json_path,
            "img_path": img_path,
            "img_name": img_name,
            "img_w": w,
            "img_h": h,
            "bbox": data["bbox"],
            "class": data["type"]
        })

    if not valid_data:
        raise ValueError("❌ Ни один файл не прошёл валидацию!")

    class_names = sorted(all_types)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    print(f"✅ Найдено классов: {len(class_names)} → {class_names}")
    print(f"✅ Валидных изображений: {len(valid_data)}")

    # Разделение на train/val
    random.seed(42)  # для воспроизводимости
    random.shuffle(valid_data)
    split_idx = int(len(valid_data) * split_ratio)
    train_data = valid_data[:split_idx]
    val_data = valid_data[split_idx:]

    def save_subset(data_list, subset):
        """Сохраняет подмножество данных (train или val) в формате YOLO."""
        (output_dir / 'images' / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / subset).mkdir(parents=True, exist_ok=True)

        for item in data_list:
            # Копируем изображение
            shutil.copy(item["img_path"], output_dir / 'images' / subset / item["img_name"])

            # Преобразуем bbox: [x1, y1, x2, y2] → нормализованный YOLO-формат
            x1, y1, x2, y2 = item["bbox"]
            img_w, img_h = item["img_w"], item["img_h"]

            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            # Проверка корректности нормализованных координат
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                print(f"⚠️ Некорректный bbox в {item['json_path']}: "
                      f"нормализованный ({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
                continue

            class_id = class_to_id[item["class"]]
            label_path = output_dir / 'labels' / subset / (Path(item["img_name"]).stem + ".txt")
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Сохраняем train и val
    save_subset(train_data, "train")
    save_subset(val_data, "val")

    # Создаём dataset.yaml с относительным путём
    yaml_content = f"""path: ./yolo_dataset
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
    with open(output_dir / "dataset.yaml", "w", encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n🎉 Конвертация завершена!")
    print(f"📁 Датасет сохранён в: {output_dir.absolute()}")
    print(f"📊 Train: {len(train_data)} изображений")
    print(f"📊 Val: {len(val_data)} изображений")


# === НАСТРОЙКИ ===
IMAGES_DIR = "data/resized_yolo"   # Папка с изображениями (например, после ресайза)
JSONS_DIR = "data/json"           # Папка с вашими JSON-аннотациями
OUTPUT_DIR = "yolo_dataset"       # Выходная директория для YOLO-датасета

if __name__ == "__main__":
    custom_json_to_yolo(IMAGES_DIR, JSONS_DIR, OUTPUT_DIR, split_ratio=0.8)


