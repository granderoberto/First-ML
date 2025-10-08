# Importiamo BaseModel e Field da Pydantic
# Pydantic serve per la validazione automatica dei dati in FastAPI
from pydantic import BaseModel, Field

# Importiamo alcuni tipi standard di Python per maggiore chiarezza
from typing import Dict, Any, List, Optional


# ================================
# 🔹 MODELLO DI RICHIESTA (INPUT)
# ================================
class PredictRequest(BaseModel):
    """
    Rappresenta la struttura del corpo della richiesta (JSON)
    che il frontend invia all'endpoint /api/predict.

    Esempio:
    {
        "features": {
            "Age": 16,
            "Gender": "Male",
            "BMI": 21.5,
            ...
        }
    }
    """

    # 'features' è un dizionario flessibile:
    # chiave = nome della colonna (es. "Age")
    # valore = input inserito dall'utente
    features: Dict[str, Any] = Field(
        ...,  # il campo è obbligatorio
        description="Coppie colonna->valore per una singola riga di input"
    )


# ================================
# 🔹 MODELLO DI RISPOSTA (OUTPUT)
# ================================
class PredictResponse(BaseModel):
    """
    Descrive il formato della risposta JSON
    che il backend restituisce dopo la predizione.

    Esempio:
    {
        "prediction": "High",
        "proba": {"Low": 0.05, "Medium": 0.10, "High": 0.85},
        "used_features": ["Age", "Gender", "BMI", ...],
        "message": "OK"
    }
    """

    # Valore della predizione (può essere numerico o testuale)
    prediction: Any

    # Probabilità per ciascuna classe (se il modello supporta predict_proba)
    proba: Optional[Dict[str, float]] = None

    # Lista dei nomi di feature effettivamente usate dal modello
    used_features: List[str]

    # Messaggio di stato (es. "OK" o descrizione errore)
    message: str