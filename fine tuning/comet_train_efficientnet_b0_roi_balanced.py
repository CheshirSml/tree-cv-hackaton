"""
ROI-классификация пород деревьев с дообучением EfficientNet-B0 и поддержкой дисбаланса классов

Назначение:
  Дообучает модель EfficientNet-B0 на сильно несбалансированном датасете ROI.
  Включает стратегии борьбы с дисбалансом: взвешенный loss, WeightedRandomSampler,
  fallback на random split при невозможности стратификации, и использование
  balanced accuracy / macro-F1 вместо обычной accuracy.

Входные данные:
  - Папка с изображениями (например, ./rois_breed/)
  - CSV-файл с колонками: 'file', 'breed'

Выход:
  - Файл с весами лучшей модели (по balanced accuracy)
  - Лог эксперимента в Comet.ml с расширенными метриками
"""

from comet_ml import Experiment
import os
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler  
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import numpy as np


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
# 3. ОБУЧЕНИЕ С COMET.ML И ПОДДЕРЖКОЙ ДИСБАЛАНСА
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
    experiment = Experiment(
        api_key=comet_api_key,
        project_name=comet_project_name,
        auto_output_logging="simple",
        auto_metric_logging=True,
        auto_param_logging=True,
        log_code=True
    )

    hyper_params = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "model": "efficientnet_b0",
        "pretrained": True,
        "img_size": 224,
        "sampler": "WeightedRandomSampler",
        "loss": "weighted CrossEntropy"
    }
    experiment.log_parameters(hyper_params)

    # Трансформации
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Загрузка полного датасета
    full_dataset = ROIBreedDataset(root=dataset_path, csv_file=csv_path, transform=train_transform)
    num_classes = len(full_dataset.classes)
    experiment.log_other("num_classes", num_classes)
    experiment.log_other("classes", full_dataset.classes)

    # === Разбиение: стратификация или fallback ===
    all_labels = full_dataset.df['label'].values
    indices = np.arange(len(all_labels))

    min_class_count = np.min(np.bincount(all_labels))
    if min_class_count >= 2:
        print("✅ Используем стратифицированное разбиение")
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=all_labels,
            random_state=42
        )
    else:
        print("⚠️ Некоторые классы имеют <2 образцов → используем random split")
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        train_idx = train_dataset.indices
        val_idx = val_dataset.indices

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = val_transform

    # === WeightedRandomSampler ===
    train_labels = [full_dataset.df.iloc[i]['label'] for i in train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)  # избегаем деления на 0
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    # Взвешенный loss
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Логирование архитектуры (с защитой)
    try:
        experiment.set_model_graph(model)
    except Exception as e:
        print(f"⚠️ Не удалось залогировать граф модели: {e}")

    # Все возможные метки для корректных метрик
    all_labels_for_metrics = np.arange(num_classes)

    # Обучение
    best_balanced_acc = 0.0
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
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)

        val_acc = np.mean(val_preds == val_targets)
        val_balanced_acc = balanced_accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(
            val_targets, val_preds,
            average='macro',
            labels=all_labels_for_metrics,
            zero_division=0
        )

        scheduler.step()

        # Логирование
        experiment.log_metrics({
            "train_loss": train_loss_avg,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_balanced_acc": val_balanced_acc,
            "val_macro_f1": val_macro_f1,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, epoch=epoch)

        print(f"Epoch {epoch+1}: "
              f"Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}, "
              f"Balanced Acc={val_balanced_acc:.4f}, "
              f"Macro F1={val_macro_f1:.4f}")

        # Сохранение по balanced accuracy
        if val_balanced_acc > best_balanced_acc:
            best_balanced_acc = val_balanced_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': full_dataset.classes,
                'label_encoder': full_dataset.label_encoder,
                'epoch': epoch,
                'val_balanced_acc': val_balanced_acc,
                'val_macro_f1': val_macro_f1
            }, save_path)
            experiment.log_model("best_model", save_path)
            print(f"✅ Новая лучшая модель сохранена (balanced acc={best_balanced_acc:.4f})")

            # Classification report — безопасно
            report = classification_report(
                val_targets,
                val_preds,
                labels=all_labels_for_metrics,
                target_names=full_dataset.classes,
                zero_division=0
            )
            experiment.log_text(f"Classification Report (Epoch {epoch}):\n{report}")

    experiment.log_other("best_val_balanced_acc", best_balanced_acc)
    experiment.end()
    print(f"\nОбучение завершено. Лучшая balanced accuracy: {best_balanced_acc:.4f}")


# ==============================
# 4. ЗАПУСК
# ==============================

if __name__ == "__main__":
    ROI_DIR = "breed_classification_dataset/rois_breed"
    CSV_FILE = "breed_classification_dataset/labels.csv"
    MODEL_SAVE_PATH = "_models/best_efficientnet_b0_breed_comet_balanced.pth"

    COMET_API_KEY = ""  # ← замените на свой!
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