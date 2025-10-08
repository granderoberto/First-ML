# PE Performance Predictor (Locale)

## Requisiti

- Python 3.10+ (consigliato)
- pip
- Visual Studio/VS Code

## Setup

```bash
cd ml-pe-app
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
```


# 🏋️‍♂️ PE Performance Predictor

Web app che utilizza un modello di Machine Learning (Random Forest)
per predire la performance in Educazione Fisica in base a dati studente.

## ⚙️ Come eseguire

1. Clona il repo:
   ```bash
   git clone https://github.com/granderoberto/First-ML.git
   cd First-ML
   uvicorn backend.app:app --reload
   ```
