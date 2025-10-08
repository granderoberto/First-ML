# Import delle librerie principali di FastAPI
from fastapi import FastAPI, Body, HTTPException  # FastAPI per creare l'app, Body per leggere i dati del corpo delle richieste, HTTPException per gestire errori API
# Middleware CORS per permettere richieste da origini diverse (es. frontend su localhost)
from fastapi.middleware.cors import CORSMiddleware
# StaticFiles serve per servire file statici (HTML, CSS, JS) del frontend
from fastapi.staticfiles import StaticFiles
# Tipi Python per annotare i parametri e migliorare la leggibilità
from typing import Dict, Any
# Modulo standard per gestire i percorsi dei file
import os
# Import della classe che gestisce il modello di Machine Learning
from .model_service import ModelService
# Import di NumPy per gestire tipi numerici e conversioni
import numpy as np
import re

app = FastAPI(title="PE Performance Predictor", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

model_service = ModelService()




# ------------------- CREAZIONE DELL'APPLICAZIONE FASTAPI -------------------

# Inizializza l'app FastAPI e imposta alcune info di descrizione
app = FastAPI(title="PE Performance Predictor", version="1.0")


# ------------------- CONFIGURAZIONE DEL MIDDLEWARE CORS -------------------

# Aggiunge un middleware per gestire richieste HTTP provenienti da altri domini
# Serve a evitare errori CORS (Cross-Origin Resource Sharing) durante lo sviluppo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # consente richieste da qualsiasi dominio
    allow_credentials=True,       # consente l’invio di cookie o header di autenticazione
    allow_methods=["*"],          # consente tutti i metodi HTTP (GET, POST, PUT, DELETE, ecc.)
    allow_headers=["*"],          # consente tutti gli header personalizzati
)


# ------------------- INIZIALIZZAZIONE DEL MODELLO -------------------

# Crea un'istanza della classe ModelService, che carica il modello ML,
# gli encoder e lo scaler dai file .pkl presenti nella cartella "data/"
model_service = ModelService()


# ------------------- ENDPOINT /api/schema -------------------

# Endpoint GET per ottenere lo "schema" dei campi richiesti dal modello
# (cioè: nomi delle colonne, tipo di campo e possibili valori per i campi categorici)
@app.get("/api/schema")
def get_schema():
    # Ritorna al frontend lo schema dei dati attesi per la predizione
    return model_service.schema_for_frontend()


# ------------------- ENDPOINT /api/predict -------------------

# Endpoint POST per ricevere i dati dallo user (frontend) e restituire una predizione
@app.post("/api/predict")
def predict(payload: Dict[str, Any] = Body(...)):
    try:
        # 1️⃣ Recupera il dizionario di feature dal body della richiesta
        # Il frontend può inviare {"features": {...}} oppure direttamente {...}
        features = payload.get("features", payload)

        # 2️⃣ Esegue la predizione usando il servizio del modello ML
        pred, proba, used = model_service.predict_one(features)

        # 3️⃣ Converte la predizione in tipo Python puro (es. int invece di numpy.int64)
        if isinstance(pred, np.generic):
            pred = pred.item()

        model_name = getattr(getattr(model_service.model, "__class__", None), "__name__", "UnknownModel")

        # 4️⃣ Ritorna il risultato al frontend in formato JSON
        return {
            "prediction": pred,       # valore predetto (es. "High" o 2)
            "proba": proba,           # dizionario di probabilità per ogni classe
            "used_features": used,    # elenco delle feature usate dal modello
            "model_name": model_name,     # 👈 Aggiunto
            "message": "OK"           # messaggio di conferma

        }

    # ------------------- GESTIONE ERRORI -------------------
    except Exception as e:
        import traceback
        # Stampa il traceback completo in console (utile per debug)
        traceback.print_exc()

        # Restituisce un errore HTTP 500 al frontend con il messaggio dell’eccezione
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/api/parse_text")
def parse_text(payload: Dict[str, str] = Body(...)):
    """
    Riceve una frase naturale e prova a generare un dizionario di feature.
    """
    text = payload.get("text", "").lower()

    features = {}

    # Esempio di estrazioni semplici con regex o keyword matching
    # (puoi arricchirlo con spaCy o un modello LLM se vuoi più precisione)
    age_match = re.search(r"(\d{1,2})\s*anni", text)
    if age_match:
        features["Age"] = int(age_match.group(1))

    if "maschio" in text or "uomo" in text:
        features["Gender"] = "Male"
    elif "femmina" in text or "donna" in text:
        features["Gender"] = "Female"
    else:
        features["Gender"] = "Other"

    if "motivata" in text or "motivato" in text or "alta motivazione" in text:
        features["Motivation_Level"] = "High"
    elif "poca motivazione" in text or "bassa motivazione" in text:
        features["Motivation_Level"] = "Low"
    else:
        features["Motivation_Level"] = "Medium"

    # Esempio per ore di attività
    match_hours = re.search(r"(\d{1,2})\s*(ore|volte)", text)
    if match_hours:
        features["Hours_Physical_Activity_Per_Week"] = int(match_hours.group(1))
    else:
        features["Hours_Physical_Activity_Per_Week"] = 3

    # Esempio di partecipazione
    if "partecipa sempre" in text or "molto attiva" in text:
        features["Class_Participation_Level"] = "High"
    elif "poco partecipe" in text:
        features["Class_Participation_Level"] = "Low"
    else:
        features["Class_Participation_Level"] = "Medium"

    # Default numerici per campi non citati
    defaults = {
        "Strength_Score": 75,
        "Endurance_Score": 70,
        "Flexibility_Score": 60,
        "Speed_Agility_Score": 65,
        "BMI": 22.5,
        "Health_Fitness_Knowledge_Score": 70,
        "Skills_Score": 75,
        "Attendance_Rate": 95,
        "Overall_PE_Performance_Score": 80,
        "Improvement_Rate": 10,
        "Final_Grade": "B",
        "Previous_Semester_PE_Grade": "B",
        "Grade_Level": "11th",
    }

    for k, v in defaults.items():
        if k not in features:
            features[k] = v

    return {"features": features, "message": "Features generate dal testo con successo!"}


# ------------------- SERVE IL FRONTEND STATICO -------------------

# Determina il percorso assoluto della cartella principale del progetto
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Costruisce il percorso assoluto della cartella "frontend"
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Monta la cartella frontend come root del sito web ("/")
# In questo modo FastAPI servirà automaticamente index.html, script.js, styles.css, ecc.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")