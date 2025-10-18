import os
import math
import json
from PIL import Image
# import google.generativeai as genai
from ultralytics import YOLO
import torch

def analyze_plant_image(image: Image.Image, yolo_model_path: str = "yolov8n.pt"):
    """
    Анализирует фото растения: детекция, отдаёт bbox`ы
    :param image: обследуемое фото
    :yolo_model_path: путь к обученной модели
    :return: bbox[]
    """

    # Запуск YOLO
    yolo_model = YOLO(yolo_model_path)
    return yolo_model(image, verbose=False)

def get_prediction_boxes(results):
    # Сбор всех bounding boxes с confidence >= 0.5
    all_boxes = []
    for result in results:
        for box in result.boxes:
            if box.conf >= 0.5:
                all_boxes.append(box.xyxy[0].tolist())

    return all_boxes

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

def find_box_result(results, target_bbox, eps=1e-3):
    target_tensor = torch.tensor(target_bbox, dtype=torch.float32)

    for result in results:
        for box in result.boxes:
            box_tensor = box.xyxy[0]
            if torch.allclose(box_tensor, target_tensor, atol=eps):
                return result

    return None