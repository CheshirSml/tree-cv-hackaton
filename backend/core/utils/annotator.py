# utils/annotator.py
import os
import uuid
from PIL import Image
from io import BytesIO
from django.core.files.base import ContentFile
import cv2
from transliterate import translit


import numpy as np

def draw_single_detection(result, index=0):
    """
    Отрисовывает только один объект YOLO (bbox + сегментация).
    """
    img = result.orig_img.copy()

    # --- 1. Достаём бокс ---
    box = result.boxes.xyxy[index].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 205), 3)

    # --- 2. Подпись (класс и confidence) ---
    cls = int(result.boxes.cls[index])
    conf = float(result.boxes.conf[index])
    label = translit(f"{result.names[cls]} {conf:.2f}", 'ru', reversed=True)
    cv2.putText(img, label, (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 205), 2)

    # --- 3. Сегментация (если есть) ---
    if result.masks is not None:
        mask = result.masks.data[index].cpu().numpy()  # [H, W]
        color = np.array([255, 0, 205], dtype=np.uint8)  # зелёный
        img[mask.astype(bool)] = img[mask.astype(bool)] * 0.5 + color * 0.5

    return img

def annotate_photo(photo_obj, primary_bbox, results):
    """
    Сохраняет на фото отметку YOLO и сохраняет результат в photo.annotation.annotated_photo.
    """

    # пример использования:
    result = results[0]  # берём первый результат
    result = results[0]
    index = 0
    bboxes = result.boxes.xyxy.cpu().numpy()

    # ищем бокс по координатам
    idx = np.where((np.abs(bboxes - primary_bbox) < 1e-2).all(axis=1))[0]
    if len(idx):
        index = idx[0]

    annotated_img = draw_single_detection(result, index=index)
    im_pil = Image.fromarray(annotated_img[:, :, ::-1])

    # 4. Генерируем имя файла
    base, ext = os.path.splitext(os.path.basename(photo_obj.photo.name))
    unique_suffix = uuid.uuid4().hex[:8]  # короткий UUID
    filename = f"{base}-annotated-{unique_suffix}{ext}"

    # 5. Сохраняем
    buffer = BytesIO()
    im_pil.save(buffer, format="JPEG")
    file_content = ContentFile(buffer.getvalue(), name=filename)

    photo_obj.annotation.annotated_photo.save(filename, file_content, save=True)

    return photo_obj.annotation.annotated_photo.url
