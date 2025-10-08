# Import base di Python e librerie scientifiche
import json                              # (non usato direttamente ma utile per debug o serializzazione)
from typing import Dict, Any, List, Tuple, Optional  # Tipi per annotazioni chiare e strutturate
import numpy as np                       # Gestione array e numeri
import pandas as pd                      # Gestione tabelle e DataFrame
from joblib import load                  # Carica modelli o scaler salvati con joblib
import pickle                            # Alternativa per caricare oggetti Python serializzati
import os                                # Per lavorare con percorsi di file e directory


# -------------------- PERCORSI FILE --------------------
# Calcola il percorso della cartella "data" rispetto al file corrente
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Percorsi assoluti dei tre artefatti fondamentali
MODEL_PATH = os.path.join(DATA_DIR, "random_forest_model.pkl")   # modello ML
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")               # scaler dei dati
ENCODERS_PATH = os.path.join(DATA_DIR, "label_encoders.pkl")     # encoders per le variabili categoriche


# -------------------- CLASSE PRINCIPALE --------------------
class ModelService:
    """
    Questa classe incapsula tutta la logica per:
    - caricare il modello ML e i suoi oggetti di preprocessing
    - preparare i dati di input
    - fare la predizione
    - restituire risultati e schema per il frontend
    """

    def __init__(self) -> None:
        """Costruttore: carica modello, scaler e label encoders."""

        # Carica gli oggetti salvati (prima prova joblib, poi pickle)
        self.model = self._safe_load(MODEL_PATH)
        self.scaler = self._safe_load(SCALER_PATH)
        self.encoders = self._safe_load(ENCODERS_PATH)

        # --- Recupero dei nomi delle feature ---
        # Alcuni modelli scikit-learn salvano 'feature_names_in_' come array numpy
        self.model_feature_names = getattr(self.model, "feature_names_in_", None)
        if isinstance(self.model_feature_names, np.ndarray):
            self.model_feature_names = self.model_feature_names.tolist()

        # Anche lo scaler può avere 'feature_names_in_'
        self.scaler_feature_names = getattr(self.scaler, "feature_names_in_", None)
        if isinstance(self.scaler_feature_names, np.ndarray):
            self.scaler_feature_names = self.scaler_feature_names.tolist()

        # Numero di feature attese dal modello (serve per verifiche)
        self.model_n_features = getattr(self.model, "n_features_in_", None)
        if isinstance(self.model_n_features, (np.ndarray, list, tuple)):
            self.model_n_features = int(self.model_n_features[0])
        elif self.model_n_features is not None:
            self.model_n_features = int(self.model_n_features)

        # --- Opzioni categoriche per il frontend ---
        # Es. "Gender": ["Male", "Female", "Other"]
        self.categorical_options: Dict[str, List[str]] = {}
        if isinstance(self.encoders, dict):
            for col, enc in self.encoders.items():
                # Se è un LabelEncoder vero e proprio
                if hasattr(enc, "classes_"):
                    self.categorical_options[col] = [str(x) for x in enc.classes_]
                # Se è un dizionario (mapping esplicito)
                elif isinstance(enc, dict):
                    self.categorical_options[col] = [str(x) for x in enc.keys()]
                # Se è una lista o array
                elif isinstance(enc, (list, tuple, np.ndarray)):
                    self.categorical_options[col] = [str(x) for x in enc]
                else:
                    self.categorical_options[col] = []

        # Deduce la lista definitiva delle feature che il modello si aspetta
        self.expected_feature_names = self._infer_feature_names()


    # =========================================================
    # ---------------------- METODI UTILI ----------------------
    # =========================================================

    def _safe_load(self, path: str):
        """Carica un file salvato con joblib o pickle (fallback sicuro)."""
        try:
            return load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)


    def _infer_feature_names(self) -> List[str]:
        """
        Tenta di determinare quali colonne (feature) il modello si aspetta in input.
        Strategia:
        1️⃣ Usa direttamente feature_names_in_ se disponibile.
        2️⃣ Altrimenti, se scaler e modello hanno stesso numero di feature, usa quelle dello scaler.
        3️⃣ Se mancano entrambi, unisce nomi da encoder e scaler e rimuove i campi di target/id.
        """

        # Caso ideale: il modello conosce i nomi
        if self.model_feature_names is not None:
            return list(self.model_feature_names)

        # Altrimenti prova con i nomi dello scaler
        if self.scaler_feature_names is not None and self.model_n_features is not None:
            if len(self.scaler_feature_names) == int(self.model_n_features):
                return list(self.scaler_feature_names)

        # Fallback: unione nomi da encoders e scaler
        names = set()
        if isinstance(self.encoders, dict):
            names.update(self.encoders.keys())
        if self.scaler_feature_names is not None:
            names.update(self.scaler_feature_names)

        # Esclude colonne che rappresentano target, ID o label
        blacklist = {
            "id", "target", "label", "labels", "y",
            "performance", "performance_label", "performance_labels",
            "performance_dummy", "performance_dummy_labels",
            # Se "overall_pe_performance_score" era il target, puoi escluderlo da qui
            # "overall_pe_performance_score",
        }

        # Pulisce la lista finale
        clean = [c for c in names if c and c.lower() not in blacklist]

        # Se conosciamo quante feature servono, taglia o ordina
        if self.model_n_features is not None:
            n = int(self.model_n_features)
            if len(clean) > n:
                prefer = list(self.scaler_feature_names) if self.scaler_feature_names is not None else []
                ordered = [c for c in prefer if c in clean] + sorted([c for c in clean if c not in prefer])
                clean = ordered[:n]

        return clean


    # =========================================================
    # ---------------------- API FRONTEND ----------------------
    # =========================================================

    def schema_for_frontend(self) -> Dict[str, Any]:
        """
        Costruisce la "scheda" (schema) che descrive al frontend quali campi mostrare nel form.
        Ogni campo contiene nome, tipo (numero o select), e eventuali opzioni.
        """
        feature_names = list(self.expected_feature_names)

        fields = []
        for name in feature_names:
            if name in self.categorical_options and self.categorical_options[name]:
                # Campo di tipo "select" con opzioni note
                fields.append({"name": name, "type": "select", "options": self.categorical_options[name]})
            else:
                # Campo numerico standard
                fields.append({"name": name, "type": "number"})

        # Restituisce lo schema completo
        return {
            "fields": fields,
            "note": f"{len(feature_names)} campi esposti. L’ordine segue quello atteso dal modello.",
        }


    # =========================================================
    # ---------------------- PREPROCESS ----------------------
    # =========================================================

    def _safe_label_transform(self, col: str, value: Any) -> Any:
        """
        Converte un valore categorico nel corrispondente valore numerico,
        usando l'encoder salvato. Gestisce vari casi (LabelEncoder, dict, lista).
        """
        enc = self.encoders.get(col) if isinstance(self.encoders, dict) else None
        if enc is None:
            return value

        val = str(value)

        if hasattr(enc, "transform"):
            # Caso LabelEncoder
            try:
                return int(enc.transform([val])[0])
            except Exception:
                # Se valore non visto, prova ad aggiungerlo alle classi conosciute
                if hasattr(enc, "classes_"):
                    classes = list(enc.classes_)
                    if val not in classes:
                        classes.append(val)
                        enc.classes_ = np.array(classes, dtype=object)
                    return int(enc.transform([val])[0])
                return -1

        # Caso dizionario semplice
        if isinstance(enc, dict):
            return int(enc.get(val, -1))

        # Caso lista o array
        if isinstance(enc, (list, tuple, np.ndarray)):
            try:
                return int(list(enc).index(val))
            except ValueError:
                return -1

        return value


    def _align_and_scale(self, X: pd.DataFrame) -> np.ndarray:
        """
        Allinea le colonne dell'input con l'ordine atteso dal modello
        e applica lo scaler solo dove serve.
        """
        # Allineamento colonne all’ordine del modello
        if self.model_feature_names is not None:
            for col in self.model_feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X.loc[:, self.model_feature_names]
        else:
            exp = list(self.expected_feature_names)
            for col in exp:
                if col not in X.columns:
                    X[col] = 0
            X = X.loc[:, exp]

        # Scaling dei valori numerici
        if self.scaler is not None and hasattr(self.scaler, "transform"):
            if self.scaler_feature_names is not None and len(self.scaler_feature_names) > 0:
                cols = list(self.scaler_feature_names)
                for c in cols:
                    if c not in X.columns:
                        X[c] = 0.0
                X[cols] = self.scaler.transform(X[cols])
            else:
                num_cols = X.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    X[num_cols] = self.scaler.transform(X[num_cols])

        # Ritorna i valori come array numpy pronto per il modello
        return X.values


    # =========================================================
    # ---------------------- PREDICT ----------------------
    # =========================================================

    def predict_one(self, features: Dict[str, Any]) -> Tuple[Any, Optional[Dict[str, float]], List[str]]:
        """
        Esegue una singola predizione a partire da un dizionario di feature.
        Restituisce: (predizione, probabilità, feature usate)
        """
        # 1️⃣ Crea un DataFrame con una sola riga
        df = pd.DataFrame([features])

        # 2️⃣ Applica encoding alle colonne categoriche conosciute
        if isinstance(self.encoders, dict):
            for col in self.encoders.keys():
                if col in df.columns:
                    df[col] = df[col].astype(str).map(lambda v: self._safe_label_transform(col, v))

        # 3️⃣ Aggiunge eventuali colonne mancanti
        exp = list(self.model_feature_names) if self.model_feature_names is not None else list(self.expected_feature_names)
        for col in exp:
            if col not in df.columns:
                df[col] = 0

        # 4️⃣ Converte in numerico dove possibile
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    df[c] = pd.to_numeric(df[c])
                except Exception:
                    if not (isinstance(self.encoders, dict) and c in self.encoders):
                        df[c] = 0

        # 5️⃣ Controlla che non restino colonne non numeriche prima dello scaling
        non_numeric_before_scale = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(non_numeric_before_scale) > 0:
            raise ValueError(
                f"Colonne non numeriche prima dello scaling: {non_numeric_before_scale}. "
                f"Valori: {df[non_numeric_before_scale].to_dict(orient='records')[0]}"
            )

        # 6️⃣ Allinea, scala e ottieni la predizione
        X = self._align_and_scale(df)

        if np.isnan(X).any():
            raise ValueError("Sono presenti NaN nell'input dopo l'allineamento/scaling. Controlla i valori inseriti.")

        y_pred = self.model.predict(X)

        # 7️⃣ Converti la predizione in tipo Python puro
        pred_value = y_pred[0]
        if isinstance(pred_value, np.generic):
            pred_value = pred_value.item()

        # 8️⃣ Calcola probabilità per ogni classe (se disponibile)
        proba_dict: Optional[Dict[str, float]] = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            classes = getattr(self.model, "classes_", None)
            if classes is None:
                classes = getattr(self.model, "classes", None)
            if isinstance(classes, np.ndarray):
                classes = classes.tolist()
            if classes is None:
                classes = list(range(len(probs)))
            proba_dict = {str(cls): float(p) for cls, p in zip(classes, probs)}

        # 9️⃣ Restituisce tutto
        used_features = list(exp)
        return pred_value, proba_dict, used_features