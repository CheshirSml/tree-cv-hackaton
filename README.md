# Tree-CV Hackaton

## Описание проекта
Экспериментальное решение задачи автоматического анализа состояния зеленых насаждений по фотографиям.  
Мы проверили гипотезу: можно ли использовать мультимодальные большие языковые модели (LLM) для описания деревьев и выявления их дефектов.  

В ходе работы:  
- Были опробованы **LLaMA, ChatGPT-4 и Gemini**.  
- Выяснилось, что LLM **слабо определяют породу** (около 10%), но **хорошо видят повреждения** (до 80%).  
- Мы разметили небольшой датасет и обучили **YOLO Segmentation** для выделения деревьев.  
- Веб-приложение собрано также в **Android-приложение** с помощью Capacitor.  

---

## Структура проекта
- backend/ # Серверная часть (Django/FastAPI)
- frontend/ # Веб-приложение (Vue + Vuetify + Vite)
- frontend/android/ # Android-обёртка через Capacitor
- mlmodels_store/ # ML модели и веса
- fine tuning/ pipeline для сегментации деревьев: от разметки до обучения 

---

## Запуск backend

```bash
source venv/bin/activate
pip3 install -r requirements.txt
```

В .env заполнить:
```env
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=
POSTGRES_HOST=

# LLM:
PROVIDER_SECRET_KEY=
PROVIDER_URL=
PROVIDER_SUBMODEL=gemini-2.5-flash-lite
```

```bash
python3 manage.py migrate
python3 manage.py runserver
```

Модели для работы на сервере складываются в: `/var/www/tree-cv-hackaton/backend/mlmodels`

---

## Запуск frontend

В .env:
```env
# VITE_BASE_URL=https://botanicpanic.pro/tapi
VITE_BASE_URL=http://127.0.0.1:8000/api
```

```bash
nvm use
npm i
npm run dev
```

Сборка:
```bash
npm run build
```
---

## Сборка Android-приложения
```bash
npx cap copy android
cd android
./gradlew clean
./gradlew assembleDebug
# ./gradlew assembleRelease
```
---

## API
- Swagger документация: http://45.89.66.66:8071/swagger/
- Публичный API: https://botanicpanic.pro/tapi
- Вебинтерфейс: http://45.89.66.66:8070/

---

## Метрики:

### YOLO:
- mAP50 0.583
- Precision 0.78
- Recall 0.5

2.84M параметров, 9.73 GFLOPs

### Gemini Flash Lite:
- Precision ~50%
- Recall ~80%
- Тестовая выборка: 40 фотографий

---

## Итоги

- YOLO Segmentation выделяет деревья.
- LLM хорошо описывает дефекты и артефакты (трещины, дупла, сломанные ветки).
- Решение экспериментальное, но показало потенциал комбинирования CV + LLM для анализа состояния зеленых насаждений.
