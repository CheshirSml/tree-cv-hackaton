"""
Скрипт для извлечения ROI стволов деревьев ТОЛЬКО с известной породой.

Цель: подготовить датасет для обучения классификатора породы.
Игнорирует все объекты с "порода = не определено".

Результат:
- Папка rois_breed/
- Файл labels.csv с колонками: file, breed
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from lxml import etree
from shapely.geometry import Polygon
from ultralytics import YOLO


# ==============================
# 1. ПАРСИНГ ТОЛЬКО ПОРОДЫ ИЗ CVAT XML
# ==============================

def parse_breeds_from_cvat(xml_path):
    """
    Извлекает только полигоны стволов с определённой породой.
    Возвращает: { "img_name.jpg": [ { "points": [...], "breed": "берёза" }, ... ] }
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()
    annotations = {}

    for image in root.findall('image'):
        img_name = Path(image.get('name')).name
        valid_trunks = []

        for poly in image.findall('polygon'):
            if poly.get('label') != 'ствол':
                continue

            # Ищем атрибут "порода"
            breed = None
            for attr in poly.findall('attribute'):
                if attr.get('name') == 'порода':
                    breed = attr.text
                    break

            # Пропускаем, если порода не определена
            if breed is None or breed == "не определено":
                continue

            # Парсим точки
            points_str = poly.get('points')
            coords = []
            for pt in points_str.split(';'):
                x, y = pt.split(',')
                coords.append([float(x), float(y)])
            points = np.array(coords)

            valid_trunks.append({
                'points': points,
                'breed': breed
            })

        if valid_trunks:
            annotations[img_name] = valid_trunks

    return annotations


# ==============================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def mask_to_polygon(mask):
    """Преобразует бинарную маску в полигон (Shapely)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 3:
        return None
    return Polygon(largest_contour.reshape(-1, 2))


def compute_iou(poly1, poly2):
    """Вычисляет IoU между двумя полигонами."""
    if not poly1 or not poly2 or not poly1.is_valid or not poly2.is_valid:
        return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return inter / union if union > 0 else 0.0
    except:
        return 0.0


def extract_roi_from_mask(image, mask, output_size=(224, 224)):
    """
    Извлекает ROI по маске: обрезает, маскирует фон, ресайзит.
    """
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    roi_img = image[y_min:y_max+1, x_min:x_max+1]
    roi_mask = mask[y_min:y_max+1, x_min:x_max+1]

    roi_masked = roi_img.copy()
    roi_masked[~roi_mask] = 0  # фон → чёрный

    return cv2.resize(roi_masked, output_size, interpolation=cv2.INTER_LINEAR)


# ==============================
# 3. ОСНОВНАЯ ФУНКЦИЯ
# ==============================

def extract_rois_for_breed(
    model_path,
    xml_path,
    img_dir,
    output_dir,
    iou_thresh=0.4,
    output_size=(224, 224)
):
    """
    Основной пайплайн: только порода → ROI.
    """
    print("Загрузка модели YOLOv11n-seg...")
    model = YOLO(model_path)

    print("Парсинг пород из CVAT XML...")
    breed_anns = parse_breeds_from_cvat(xml_path)

    output_dir = Path(output_dir)
    rois_dir = output_dir / "rois_breed"
    rois_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    roi_id = 0

    img_dir = Path(img_dir)
    total_imgs = len(breed_anns)
    print(f"Найдено {total_imgs} изображений с определённой породой.")

    for i, (img_name, trunks) in enumerate(breed_anns.items(), 1):
        print(f"[{i}/{total_imgs}] Обработка {img_name}...")

        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            continue
        h, w = orig_img.shape[:2]

        # Инференс YOLO
        results = model(str(img_path))
        if not results[0].masks:
            continue

        # Преобразуем маски YOLO в полигоны
        yolo_masks = []
        for mask_tensor in results[0].masks.data:
            mask_np = mask_tensor.cpu().numpy()
            mask_full = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            poly = mask_to_polygon(mask_full > 0.5)
            yolo_masks.append((mask_full, poly))

        # Сопоставление по IoU
        for trunk in trunks:
            cvat_poly = Polygon(trunk['points'])
            best_iou, best_mask = 0, None

            for mask_full, yolo_poly in yolo_masks:
                iou = compute_iou(cvat_poly, yolo_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_mask = mask_full

            if best_iou >= iou_thresh and best_mask is not None:
                roi = extract_roi_from_mask(orig_img, best_mask > 0.5, output_size)
                if roi is not None:
                    roi_file = f"breed_{roi_id:05d}.jpg"
                    cv2.imwrite(str(rois_dir / roi_file), roi)
                    labels.append({'file': roi_file, 'breed': trunk['breed']})
                    roi_id += 1

    # Сохранение
    if labels:
        df = pd.DataFrame(labels)
        df.to_csv(output_dir / "labels.csv", index=False, encoding='utf-8')
        print(f"\n✅ Сохранено {len(labels)} ROI с известной породой.")
        print(f"📁 ROI: {rois_dir}/")
        print(f"📊 Метки: {output_dir}/labels.csv")
        
        # Вывод статистики по породам
        print("\nСтатистика по породам:")
        print(df['breed'].value_counts().to_string())
    else:
        print("❌ Не найдено ни одного ствола с определённой породой.")


# ==============================
# 4. ЗАПУСК
# ==============================

if __name__ == "__main__":
    # === НАСТРОЙТЕ ПУТИ ===
    MODEL_PATH = "_ROI extractor/yolo-segmentation build 2.8.pt"               # ваша YOLOv11n-seg модель
    XML_PATH = "_ROI extractor/CVATforIMAGES_part_9/annotations.xml"         # экспорт из CVAT
    IMG_DIR = "_ROI extractor/CVAT backup part_9/data"                   # папка с изображениями
    OUTPUT_DIR = "breed_classification_dataset"  # выходная папка

    extract_rois_for_breed(
        model_path=MODEL_PATH,
        xml_path=XML_PATH,
        img_dir=IMG_DIR,
        output_dir=OUTPUT_DIR,
        iou_thresh=0.4,
        output_size=(224, 224)
    )