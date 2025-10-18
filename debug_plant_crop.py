import os
import math
from PIL import Image, ImageDraw
from ultralytics import YOLO

def crop_with_padding(image: Image.Image, bbox, padding_ratio=0.1):
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    W, H = image.size
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(W, x2 + pad_x)
    y2_pad = min(H, y2 + pad_y)
    return image.crop((x1_pad, y1_pad, x2_pad, y2_pad))


def select_primary_object(boxes, image_size):
    if not boxes:
        return None
    W, H = image_size
    max_dist = math.sqrt((W / 2) ** 2 + (H / 2) ** 2)
    best_box, best_score = None, -1
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        norm_area = area / (W * H + 1e-6)
        box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.sqrt((box_cx - W / 2) ** 2 + (box_cy - H / 2) ** 2)
        norm_dist = 1 - (dist / (max_dist + 1e-6))
        score = 0.6 * norm_area + 0.4 * norm_dist
        if score > best_score:
            best_score = score
            best_box = bbox
    return best_box


def debug_plant_crop(image_path: str, yolo_model_path: str = "_final_project/tree-cv-hackaton/mlmodels_store/yolo-segmentation build 2.8.pt", output_dir: str = "output"):
    """
    Детектирует растения, выбирает главное, сохраняет его crop и возвращает данные.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка изображения
    full_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(full_image)

    # YOLO детекция
    yolo_model = YOLO(yolo_model_path)
    results = yolo_model(full_image, verbose=False)

    # Сбор bounding boxes
    all_boxes = []
    for result in results:
        for box in result.boxes:
            if box.conf >= 0.5:
                bbox = box.xyxy[0].tolist()
                all_boxes.append(bbox)
                # Опционально: нарисовать все боксы на исходном изображении
                x1, y1, x2, y2 = map(int, bbox)
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)

    if not all_boxes:
        print("❌ Нет обнаруженных растений с confidence ≥ 0.5")
        return {"error": "no_plants_detected"}

    # Выбор главного
    primary_bbox = select_primary_object(all_boxes, full_image.size)
    if primary_bbox is None:
        print("❌ Не удалось выбрать главное растение")
        return {"error": "no_primary_selected"}

    # Обрезка
    cropped_img = crop_with_padding(full_image, primary_bbox, padding_ratio=0.1)

    # Сохранение
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_crop_path = os.path.join(output_dir, f"{base_name}_primary_crop.jpg")
    output_full_path = os.path.join(output_dir, f"{base_name}_with_boxes.jpg")

    cropped_img.save(output_crop_path)
    full_image.save(output_full_path)

    # Показ изображения (только если запускаете локально с GUI)
    try:
        cropped_img.show(title="Главное растение (crop)")
    except Exception:
        pass  # Игнорировать, если нет GUI (например, на сервере)

    print(f"✅ Crop сохранён: {output_crop_path}")
    print(f"🖼️ Исходное с боксами: {output_full_path}")

    return {
        "primary_bbox": primary_bbox,
        "crop_path": output_crop_path,
        "annotated_image_path": output_full_path,
        "total_detections": len(all_boxes)
    }


# === Пример использования ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Использование: python debug_plant_crop.py <путь_к_фото.jpg>")
        sys.exit(1)

    result = debug_plant_crop(sys.argv[1])
    print("\n📊 Результат:")
    for k, v in result.items():
        print(f"  {k}: {v}")