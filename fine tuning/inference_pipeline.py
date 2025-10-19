"""
ПОЛНЫЙ СЦЕНАРИЙ ИНФЕРЕНСА: YOLOv10-seg + выбор главного ствола + EfficientNet-B0

Цель: 
  По входному изображению леса/парка определить породу главного дерева и вернуть:
    - аннотированное изображение с bbox главного ствола,
    - название породы и уверенность модели,
    - промежуточные данные для передачи в Gemini 2.5 Flash.

Этапы пайплайна (согласно заданию):
1.0. YOLOv10-seg детектирует и сегментирует все стволы.
2.0. Выбирается "главный" ствол по комбинированному критерию (размер + центральность).
3.0. По маске главного ствола извлекается ROI (Region of Interest) — изображение коры без фона.
4.0. EfficientNet-B0 классифицирует ROI и определяет породу.
5.1. Возвращается исходное изображение с красным bbox вокруг главного ствола.
5.2. Возвращается результат EfficientNet-B0: порода + уверенность (для Gemini).
5.3. Возвращаются все промежуточные данные (bbox, ROI, маска и т.д.) — для тестирования и отладки.
"""

# Импорты стандартных библиотек
import os
import math
import cv2
import numpy as np
from pathlib import Path

# Импорты для работы с изображениями и визуализацией
from PIL import Image, ImageDraw

# Импорты для моделей и инференса
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO


# ==============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def select_primary_object(boxes, image_size):
    """
    Этап 2.0: Выбор главного объекта из списка bounding boxes.
    
    Логика выбора:
      - Чем больше площадь bbox — тем выше приоритет (ствол крупнее → ближе/важнее).
      - Чем ближе центр bbox к центру изображения — тем выше приоритет (фокус съёмки).
    
    Возвращает: координаты bbox главного объекта [x1, y1, x2, y2] или None.
    """
    if not boxes:
        return None

    W, H = image_size
    max_dist = math.sqrt((W / 2) ** 2 + (H / 2) ** 2)  # максимальное расстояние до угла
    best_box, best_score = None, -1

    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        norm_area = area / (W * H + 1e-6)  # нормализуем по площади изображения

        box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.sqrt((box_cx - W / 2) ** 2 + (box_cy - H / 2) ** 2)
        norm_dist = 1 - (dist / (max_dist + 1e-6))  # 1 = в центре, 0 = в углу

        # Комбинированный скор: 60% — размер, 40% — центральность
        score = 0.6 * norm_area + 0.4 * norm_dist

        if score > best_score:
            best_score = score
            best_box = bbox

    return best_box


def extract_roi_from_mask(image, mask, output_size=(224, 224)):
    """
    Этап 3.0: Извлечение ROI по бинарной маске.
    
    Что делает:
      - Находит bounding box маски,
      - Обрезает изображение по этому bbox,
      - Применяет маску: всё, что вне ствола — становится чёрным (фон убирается),
      - Ресайзит до фиксированного размера (224×224), подходящего для EfficientNet-B0.
    
    Возвращает: numpy-массив (H, W, 3) в формате BGR или None, если маска пустая.
    """
    # Убедимся, что маска — булева (важно для совместимости с float-масками от YOLO)
    if mask.dtype != bool:
        mask = mask > 0.5

    # Найдём ненулевые пиксели
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return None

    # Определим bbox по маске
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Обрежем изображение и маску
    roi_img = image[y_min:y_max+1, x_min:x_max+1]
    roi_mask = mask[y_min:y_max+1, x_min:x_max+1]

    # Применим маску: фон → чёрный
    roi_masked = roi_img.copy()
    roi_masked[~roi_mask] = 0

    # Приведём к стандартному размеру для классификатора
    return cv2.resize(roi_masked, output_size, interpolation=cv2.INTER_LINEAR)


def create_efficientnet_model(num_classes):
    """
    Создаёт модель EfficientNet-B0 с изменённой головой под заданное число классов.
    Используется на этапе 4.0.
    """
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)  # веса загрузим позже из чекпоинта
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def predict_breed_from_roi(roi_image, model_path):
    """
    Этап 4.0: Классификация породы по ROI.
    
    Вход: 
      - roi_image: numpy-массив (BGR, H, W, 3),
      - model_path: путь к чекпоинту EfficientNet-B0 (без LabelEncoder!).
    
    Выход: (порода: str, уверенность: float)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загрузка чекпоинта (безопасно, weights_only=True)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    classes = checkpoint['classes']  # список строк: ['берёза', 'дуб', ...]
    num_classes = len(classes)

    # Инициализация и загрузка весов модели
    model = create_efficientnet_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Преобразование ROI в формат, ожидаемый EfficientNet
    roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)  # вероятности по классам
        confidence, idx = torch.max(probs, dim=1)
        breed = classes[idx.item()]

    return breed, confidence.item()


# ==============================================================================
# 2. ОСНОВНОЙ ПАЙПЛАЙН ИНФЕРЕНСА
# ==============================================================================

def run_inference_pipeline(
    image_path: str,
    yolo_model_path: str,
    efficientnet_model_path: str,
    output_dir: str = "inference_output",
    conf_threshold: float = 0.5
):
    """
    Основная функция, реализующая полный пайплайн инференса (этапы 1.0–5.3).
    
    Параметры:
      - image_path: путь к исходному изображению (.jpg, .png),
      - yolo_model_path: путь к дообученной YOLOv10-seg модели,
      - efficientnet_model_path: путь к чекпоинту EfficientNet-B0,
      - output_dir: папка для сохранения результатов,
      - conf_threshold: порог уверенности YOLO (по умолчанию 0.5).
    
    Возвращает: словарь с результатами (см. этапы 5.1–5.3).
    """
    # Создаём выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    image_path = Path(image_path)
    base_name = image_path.stem

    # --- Этап 1.0: Загрузка исходного изображения ---
    orig_cv_img = cv2.imread(str(image_path))
    if orig_cv_img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    h, w = orig_cv_img.shape[:2]

    # Конвертируем в PIL для удобной визуализации
    pil_img = Image.fromarray(cv2.cvtColor(orig_cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # --- Этап 1.0: Запуск YOLOv10-seg ---
    yolo_model = YOLO(yolo_model_path)
    results = yolo_model(pil_img, verbose=False)

    all_boxes = []
    all_masks = []

    # Собираем все детекции со степенью уверенности ≥ conf_threshold
    for result in results:
        if result.boxes is None or result.masks is None:
            continue
        for box, mask in zip(result.boxes, result.masks):
            if box.conf >= conf_threshold:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                mask_np = mask.data[0].cpu().numpy()
                # Маска от YOLO — маленькая; ресайзим до размера изображения
                mask_full = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                all_boxes.append(bbox)
                all_masks.append(mask_full)

                # Опционально: рисуем все bbox (для отладки)
                x1, y1, x2, y2 = map(int, bbox)
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)

    if not all_boxes:
        return {"error": "no_trunks_detected"}

    # --- Этап 2.0: Выбор главного ствола ---
    primary_bbox = select_primary_object(all_boxes, (w, h))
    if primary_bbox is None:
        return {"error": "no_primary_selected"}

    # --- Этап 2.0 → 3.0: Сопоставление маски главному bbox через IoU ---
    # Создаём бинарную маску bbox (для вычисления пересечения)
    x1_p, y1_p, x2_p, y2_p = primary_bbox
    primary_box_mask = np.zeros((h, w), dtype=bool)
    primary_box_mask[int(y1_p):int(y2_p), int(x1_p):int(x2_p)] = True

    # Находим маску от YOLO, которая лучше всего пересекается с bbox
    primary_mask = None
    best_iou = 0
    for mask in all_masks:
        mask_bool = mask > 0.5  # приводим float-маску к bool
        intersection = np.logical_and(mask_bool, primary_box_mask)
        union = np.logical_or(mask_bool, primary_box_mask)
        iou = np.sum(intersection) / (np.sum(union) + 1e-6)
        if iou > best_iou:
            best_iou = iou
            primary_mask = mask  # сохраняем оригинальную маску для ROI

    if primary_mask is None:
        primary_mask = all_masks[0]  # fallback

    # Рисуем bbox главного ствола красным (этап 5.1)
    x1, y1, x2, y2 = map(int, primary_bbox)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # --- Этап 3.0: Извлечение ROI по маске ---
    roi_image = extract_roi_from_mask(orig_cv_img, primary_mask > 0.5, output_size=(224, 224))
    if roi_image is None:
        return {"error": "roi_extraction_failed"}

    # --- Этап 4.0: Классификация породы ---
    breed, confidence = predict_breed_from_roi(roi_image, efficientnet_model_path)

    # --- Этапы 5.1–5.3: Сохранение и возврат результатов ---
    annotated_path = Path(output_dir) / f"{base_name}_annotated.jpg"
    roi_path = Path(output_dir) / f"{base_name}_roi.jpg"

    # Сохраняем аннотированное изображение (5.1)
    annotated_img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(annotated_path), annotated_img_bgr)
    # Сохраняем ROI (5.3)
    cv2.imwrite(str(roi_path), roi_image)

    # Возвращаем полный набор данных (5.1, 5.2, 5.3)
    return {
        # 5.1: Визуальный результат
        "annotated_image_path": str(annotated_path),
        # 5.2: Результат классификации (для Gemini)
        "breed": breed,
        "confidence": confidence,
        # 5.3: Промежуточные данные
        "roi_image_path": str(roi_path),
        "primary_bbox": primary_bbox,          # можно отправить в Gemini вместо ROI
        "primary_roi_path": str(roi_path),     # или отправить ROI
        "total_detections": len(all_boxes),
        "primary_mask": primary_mask           # полная маска (можно сохранить как .npy)
    }


# ==============================================================================
# 3. ТОЧКА ВХОДА (КОМАНДНАЯ СТРОКА)
# ==============================================================================

if __name__ == "__main__":
    import argparse

    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Инференс пайплайн: YOLOv10-seg + EfficientNet-B0 для определения породы дерева"
    )
    parser.add_argument("image", type=str, help="Путь к входному изображению")
    parser.add_argument("--yolo", type=str, default="yolov10s-seg.pt", help="Путь к модели YOLOv10-seg")
    parser.add_argument("--efficientnet", type=str, default="best_efficientnet_b0_breed_safe.pth", help="Путь к модели EfficientNet-B0")
    parser.add_argument("--output", type=str, default="inference_output", help="Папка для сохранения результатов")

    args = parser.parse_args()

    # Запуск пайплайна
    try:
        result = run_inference_pipeline(
            image_path=args.image,
            yolo_model_path=args.yolo,
            efficientnet_model_path=args.efficientnet,
            output_dir=args.output
        )

        # Вывод результата в консоль
        if "error" in result:
            print(f"❌ Ошибка: {result['error']}")
        else:
            print("✅ Инференс завершён успешно!")
            print(f"  🌳 Порода: {result['breed']}")
            print(f"  📊 Уверенность: {result['confidence']:.4f}")
            print(f"  🖼️ Аннотированное изображение: {result['annotated_image_path']}")
            print(f"  🧬 ROI: {result['roi_image_path']}")
            print(f"  📦 BBox главного ствола: {result['primary_bbox']}")
            print(f"  🔢 Всего стволов: {result['total_detections']}")

    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")