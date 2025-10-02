"""
Автоматическая генерация разметки изображений с помощью предобученной модели YOLO11-seg
и экспорт в формат COCO для последующей загрузки в CVAT (Computer Vision Annotation Tool).

Описание:
    Скрипт использует модель сегментации YOLOv8 для автоматической разметки набора изображений.
    Для каждого изображения модель предсказывает bounding boxes и сегментационные маски.
    Результаты преобразуются в стандартный формат COCO (Common Objects in Context),
    включая:
      - список изображений,
      - аннотации с полигонами сегментации и bounding boxes,
      - категории (классы) объектов.

    Полученный JSON-файл и копии изображений сохраняются в указанную директорию,
    что позволяет легко импортировать датасет в CVAT как "auto-annotated" проект.

Особенности:
    - Поддержка обрезанных (truncated) изображений благодаря `ImageFile.LOAD_TRUNCATED_IMAGES = True`.
    - Используются нормализованные полигоны из `result.masks.xyn`, которые денормализуются
      относительно размеров исходного изображения.
    - Bounding boxes берутся напрямую из предсказаний модели (`result.boxes`).
    - Пропускаются сегменты с менее чем 3 точками (некорректные полигоны).
    - Все изображения копируются в выходную папку для удобства импорта в CVAT.

Требования:
    - Установленный пакет `ultralytics`.
    - Предобученная модель сегментации (.pt), поддерживающая маски.
    - Изображения в форматах JPG/JPEG или PNG.

Структура выходной директории:
    cvat_dataset/
    ├── auto_annotations.json   # разметка в формате COCO
    ├── image1.jpg
    ├── image2.png
    └── ...

Использование:
    1. Укажите путь к модели (`best.pt`).
    2. Укажите папку с изображениями для разметки.
    3. Запустите скрипт.
    4. Импортируйте папку `cvat_dataset` в CVAT как новый датасет с разметкой.

Примечания:
    - Формат COCO, генерируемый скриптом, совместим с CVAT начиная с версии 2.0+.
    - Полигоны сохраняются как единый контур (без внутренних отверстий).
    - Параметр `iscrowd=0` указывает, что аннотация не является группой объектов.
"""

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Позволяет загружать повреждённые/обрывочные изображения

from ultralytics import YOLO
import os
from pathlib import Path
import json
from tqdm import tqdm
import shutil

# === НАСТРОЙКИ ===
# Загрузка обученной модели сегментации YOLOv8
model = YOLO('runs/segment/tree_detector2/weights/best.pt')

# Папка с новыми изображениями для автоматической разметки
images_dir = Path('Санитарка/2')
# Выходная директория для датасета в формате, пригодном для CVAT
output_dir = Path('cvat_dataset/')
output_dir.mkdir(exist_ok=True)

# Получаем список изображений (поддержка .jpg, .jpeg, .png в любом регистре)
image_files = (
    list(images_dir.glob('*.[jJ][pP][gG]')) +
    list(images_dir.glob('*.[jJ][pP][eE][gG]')) +
    list(images_dir.glob('*.[pP][nN][gG]'))
)

# Выполняем инференс моделью
print("Запуск модели для автоматической разметки...")
results = model(image_files, stream=False)  # stream=False — загружает все результаты в память сразу

# === ФОРМИРОВАНИЕ ДАТАСЕТА В ФОРМАТЕ COCO ===
coco = {
    "images": [],
    "annotations": [],
    "categories": [],
    "licenses": [],
    "info": {"description": "Auto-labeled by YOLOv8-seg"}
}

# Добавляем категории (классы) из модели
for cls_id, cls_name in model.names.items():
    coco["categories"].append({
        "id": cls_id,
        "name": cls_name,
        "supercategory": "object"
    })

annotation_id = 1  # Уникальный ID для каждой аннотации

# Обрабатываем результаты для каждого изображения
for result in tqdm(results, total=len(image_files), desc="Обработка изображений"):
    img_path = Path(result.path)
    img_name = img_path.name
    img_w, img_h = result.orig_shape[1], result.orig_shape[0]  # ширина, высота

    # Регистрируем изображение в COCO
    image_id = len(coco["images"]) + 1
    coco["images"].append({
        "id": image_id,
        "file_name": img_name,
        "width": img_w,
        "height": img_h
    })

    # Пропускаем изображения без сегментационных масок
    if result.masks is None:
        continue

    # Извлекаем данные: bounding boxes, классы и нормализованные полигоны
    boxes = result.boxes.xyxy.cpu().numpy()      # [x1, y1, x2, y2]
    classes = result.boxes.cls.cpu().numpy()     # [cls_id, ...]
    masks = result.masks.xyn                     # нормализованные полигоны: [[x1,y1,x2,y2,...], ...]

    # Обрабатываем каждую детекцию
    for i in range(len(boxes)):
        cls_id = int(classes[i])
        polygon_norm = masks[i].flatten()  # Преобразуем в плоский список: [x1, y1, x2, y2, ...]

        # Денормализуем координаты полигона до пикселей
        polygon = [
            int(coord * (img_w if j % 2 == 0 else img_h))
            for j, coord in enumerate(polygon_norm)
        ]

        # Пропускаем полигоны с менее чем 3 точками (некорректные)
        if len(polygon) < 6:
            continue

        # Извлекаем bounding box и вычисляем площадь
        x_min, y_min, x_max, y_max = boxes[i]
        w, h = x_max - x_min, y_max - y_min

        # Добавляем аннотацию в COCO-формат
        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": cls_id,
            "segmentation": [polygon],  # COCO ожидает список списков (даже для одного полигона)
            "bbox": [int(x_min), int(y_min), int(w), int(h)],
            "area": int(w * h),
            "iscrowd": 0  # 0 = одиночный объект, 1 = группа (crowd)
        })
        annotation_id += 1

# Сохраняем аннотации в JSON
with open(output_dir / "auto_annotations.json", "w", encoding='utf-8') as f:
    json.dump(coco, f, indent=2)

# Копируем все изображения в выходную директорию
print("Копирование изображений...")
for img in image_files:
    shutil.copy(img, output_dir / img.name)

print(f"✅ Датасет для CVAT успешно сохранён в: {output_dir.absolute()}")
print(f"📊 Всего обработано изображений: {len(image_files)}")
print(f"📊 Всего аннотаций создано: {len(coco['annotations'])}")