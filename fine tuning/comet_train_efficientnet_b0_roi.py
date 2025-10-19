
"""

ROI-классификация пород деревьев с дообучением EfficientNet-B0

Назначение:
  Дообучает  модель EfficientNet-B0 на датасете,
  содержащем изображения ROI (Region of Interest) и CSV-файл с аннотациями.
  Поддерживает аугментации, валидацию, сохранение лучшей модели и полное
  логирование эксперимента в Comet.ml.

Входные данные:
  - Папка с изображениями (например, ./rois_breed/)
  - CSV-файл с колонками: 'file' (имя файла) и 'breed' (метка класса)

Выход:
  - Файл с весами лучшей модели (по валидационной точности)
  - Лог эксперимента в проекте Comet.ml (метрики, гиперпараметры, архитектура, код)

Особенности:
  - Использует ImageNet-нормализацию и аугментации для устойчивости модели
  - Автоматическое разделение данных (80/20) с фиксированным seed
  - Интеграция с Comet.ml: отслеживание гиперпараметров, метрик, версий кода
  - Сохраняет вместе с моделью: список классов и LabelEncoder для инференса

"""
# Инициализация Comet.ml
from comet_ml import Experiment

import os
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm



# ==============================
# 1. ПОДГОТОВКА ДАТАСЕТА
# ==============================

class ROIBreedDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = Path(root)
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        self.label_encoder = LabelEncoder()
        self.df['label'] = self.label_encoder.fit_transform(self.df['breed'])
        self.classes = list(self.label_encoder.classes_)

        print(f"Загружено {len(self.df)} изображений")
        print("Породы:", self.classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row['file']
        image = Image.open(img_path).convert('RGB')
        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, label


# ==============================
# 2. МОДЕЛЬ
# ==============================

def create_model(num_classes, pretrained=True):
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    if pretrained:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ==============================
# 3. ОБУЧЕНИЕ С COMET.ML
# ==============================

def train_model_with_comet(
    dataset_path,
    csv_path,
    num_epochs=30,
    batch_size=32,
    lr=1e-4,
    save_path="best_efficientnet_b0_breed.pth",
    comet_api_key=None,
    comet_project_name="tree-breed-roi"
):
    # Инициализация эксперимента в Comet
    experiment = Experiment(
        api_key=comet_api_key,
        project_name=comet_project_name,
        auto_output_logging="simple",
        auto_metric_logging=True,
        auto_param_logging=True,
        log_code=True
    )

    # Гиперпараметры
    hyper_params = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "model": "efficientnet_b0",
        "pretrained": True,
        "img_size": 224
    }
    experiment.log_parameters(hyper_params)

    # Трансформации
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Датасет
    full_dataset = ROIBreedDataset(root=dataset_path, csv_file=csv_path, transform=train_transform)
    num_classes = len(full_dataset.classes)
    experiment.log_other("num_classes", num_classes)
    experiment.log_other("classes", full_dataset.classes)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Модель и оптимизатор
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Логирование архитектуры
    experiment.set_model_graph(model)

    # Обучение
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)

        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total

        # Обновление LR
        scheduler.step()

        # Логирование в Comet
        experiment.log_metrics({
            "train_loss": train_loss_avg,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, epoch=epoch)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Сохранение лучшей модели
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': full_dataset.classes,
                'label_encoder': full_dataset.label_encoder,
                'epoch': epoch,
                'val_acc': val_acc
            }, save_path)
            experiment.log_model("best_model", save_path)
            print(f"✅ Новая лучшая модель сохранена (acc={best_acc:.4f})")

    experiment.log_other("best_val_acc", best_acc)
    experiment.end()
    print(f"\nОбучение завершено. Лучшая точность: {best_acc:.4f}")


# ==============================
# 4. ЗАПУСК
# ==============================

if __name__ == "__main__":
    # === НАСТРОЙТЕ ===
    ROI_DIR = "breed_classification_dataset/rois_breed"
    CSV_FILE = "breed_classification_dataset/labels.csv"
    MODEL_SAVE_PATH = "best_efficientnet_b0_breed_comet.pth"

    # Ваш API-ключ из https://www.comet.com
    COMET_API_KEY = ""  # ← замените!
    COMET_PROJECT_NAME = "tree-breed-roi"

    train_model_with_comet(
        dataset_path=ROI_DIR,
        csv_path=CSV_FILE,
        num_epochs=30,
        batch_size=32,
        lr=1e-4,
        save_path=MODEL_SAVE_PATH,
        comet_api_key=COMET_API_KEY,
        comet_project_name=COMET_PROJECT_NAME
    )



