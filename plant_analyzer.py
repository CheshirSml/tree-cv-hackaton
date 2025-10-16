import os
import math
import json
from PIL import Image
import google.generativeai as genai
from ultralytics import YOLO

# === 1. Настройка Gemini ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Установите переменную окружения GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# === 2. Промпт для LLM ===
PROMPT = """
You are a plant pathology expert. Analyze the provided image and do the following:
1. Identify the plant species.
2. Diagnose any visible disease, pest damage, or nutrient deficiency.
3. If the plant appears healthy, state "No disease detected".

Respond ONLY in valid JSON format with these keys:
{"species": "string", "disease": "string", "confidence": "high|medium|low"}
Do not add any other text or explanation.
"""

# === 3. Вспомогательные функции ===

def crop_with_padding(image: Image.Image, bbox, padding_ratio=0.1):
    """Обрезает изображение по bounding box с отступом."""
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
    """
    Выбирает главное растение по комбинированному скору: размер + центральность.
    :param boxes: список [x1, y1, x2, y2]
    :param image_size: (ширина, высота)
    :return: bounding box главного объекта или None
    """
    if not boxes:
        return None

    W, H = image_size
    max_dist = math.sqrt((W / 2) ** 2 + (H / 2) ** 2)  # расстояние от центра до угла

    best_box, best_score = None, -1

    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        # Площадь
        area = (x2 - x1) * (y2 - y1)
        norm_area = area / (W * H + 1e-6)

        # Центр bounding box
        box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.sqrt((box_cx - W / 2) ** 2 + (box_cy - H / 2) ** 2)
        norm_dist = 1 - (dist / (max_dist + 1e-6))

        # Комбинированный скор
        score = 0.6 * norm_area + 0.4 * norm_dist

        if score > best_score:
            best_score = score
            best_box = bbox

    return best_box


# === 4. Основная функция анализа ===

def analyze_plant_image(image_path: str, yolo_model_path: str = "yolov8n.pt"):
    """
    Анализирует фото растения: детекция → выбор главного → диагностика через Gemini.
    """
    # Загрузка изображения
    full_image = Image.open(image_path).convert("RGB")

    # Запуск YOLO
    yolo_model = YOLO(yolo_model_path)
    results = yolo_model(full_image, verbose=False)

    # Сбор всех bounding boxes с confidence >= 0.5
    all_boxes = []
    for result in results:
        for box in result.boxes:
            if box.conf >= 0.5:
                all_boxes.append(box.xyxy[0].tolist())

    if not all_boxes:
        return {"error": "No plants detected with sufficient confidence."}

    # Выбор главного растения
    primary_bbox = select_primary_object(all_boxes, full_image.size)
    if primary_bbox is None:
        return {"error": "Could not select primary plant."}

    # Обрезка
    cropped_img = crop_with_padding(full_image, primary_bbox, padding_ratio=0.1)

    # Запрос к Gemini
    try:
        response = gemini_model.generate_content(
            [PROMPT, cropped_img],
            generation_config=genai.GenerationConfig(
                max_output_tokens=200,
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        # Парсинг JSON
        result = json.loads(response.text)
        return result
    except Exception as e:
        return {"error": f"LLM failed: {str(e)}"}


# === 5. Пример использования ===

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Использование: python plant_analyzer.py <путь_к_фото.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = analyze_plant_image(image_path)

    print("\n🔍 Результат анализа:")
    print(json.dumps(result, indent=2, ensure_ascii=False))