# utils/annotator.py
import os
import uuid
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from django.core.files.base import ContentFile

cv_model = YOLO("mlmodels/best_maksim-tree.pt")


def annotate_photo(photo_obj):
    """
    Берёт исходное фото (photo.photo),
    прогоняет через YOLO и сохраняет результат в photo.annotation.annotated_photo.
    """

    # 1. Загружаем исходное изображение
    img = Image.open(photo_obj.photo)

    # 2. YOLO
    results = cv_model(img)
    result = results[0]

    # 3. Отрисовка
    annotated_img = result.plot()[:, :, ::-1]  # BGR->RGB
    im_pil = Image.fromarray(annotated_img)

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
