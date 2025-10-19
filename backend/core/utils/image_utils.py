import io
import os
import base64

from uuid import uuid4
from pathlib import Path
from PIL import Image
from typing import Tuple
from io import BytesIO
from typing import Union, BinaryIO


TARGET_SIZE = (640, 640)

def get_image(file: BinaryIO):
    file_bytes = file.read()
    try:
        image = Image.open(io.BytesIO(file_bytes))
    except Exception:
        raise Exception("Некорректный формат изображения")
    return image

def get_image_size(file: BytesIO) -> Tuple[int, int]:
    image = Image.open(file)
    width, height = image.size
    file.seek(0)
    return width, height

def resize_image(image_path: Path, size: Tuple[int, int] = TARGET_SIZE) -> Path:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail(size, Image.LANCZOS)

        # Создание нового холста с белым фоном
        new_img = Image.new("RGB", size, (255, 255, 255))
        offset = (
            (size[0] - img.width) // 2,
            (size[1] - img.height) // 2
        )
        new_img.paste(img, offset)

        # Сохраняем с суффиксом
        resized_path = image_path.with_stem(image_path.stem + "_resized")
        new_img.save(resized_path, format="JPEG")

    return resized_path

def resize_to_square(image: Image.Image, size: int = 1024) -> Image.Image:
    # масштабируем картинку с сохранением пропорций
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    # создаём квадратный холст
    new_img = Image.new("RGB", (size, size), (0, 0, 0))
    # вставляем в центр
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    new_img.paste(image, offset)
    return new_img

def get_base64_image(file: Union[BinaryIO, Image.Image]):
    if isinstance(file, Image.Image):
        image = file
    else:
        file_bytes = file.read()
        try:
            image = Image.open(io.BytesIO(file_bytes))
        except Exception:
            raise Exception("Некорректный формат изображения")

    # сохраняем в JPEG в память
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)

    # кодируем в base64
    b64_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

def rotate_image_90(image: Image.Image, angle: int) -> Image.Image:
    """
    Поворачивает изображение на угол, кратный 90°.
    angle: 90, 180, 270 (по часовой стрелке)
    """
    angle = angle % 360
    if angle == 90:
        return image.transpose(Image.ROTATE_270)  # по часовой стрелке
    elif angle == 180:
        return image.transpose(Image.ROTATE_180)
    elif angle == 270:
        return image.transpose(Image.ROTATE_90)
    elif angle == 0:
        return image.copy()
    else:
        return image.copy()
    