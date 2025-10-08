// ======================================================
// 📄 script.js — logica frontend per PE Performance Predictor
// ======================================================

// Elementi principali della pagina
const formEl   = document.getElementById("predict-form"); // form dove vengono creati dinamicamente i campi
const resultEl = document.getElementById("result");       // box dei risultati
const predictBtn = document.getElementById("predict-btn"); // bottone "Predici"
const resetBtn   = document.getElementById("reset-btn");   // bottone "Reset"
const fillBtn = document.getElementById("fill-btn");       // bottone "Compila casuale"

// Endpoint API (stesso dominio di FastAPI)
const API_SCHEMA = "/api/schema";   // restituisce i campi dinamici (feature del modello)
const API_PREDICT = "/api/predict"; // endpoint per fare la predizione


// ======================================================
// 🔧 Funzione: crea un campo input o select a partire dallo schema ricevuto dal backend
// ======================================================
function createField(field) {
  const wrap = document.createElement("div");
  wrap.className = "field";

  // Etichetta visiva
  const label = document.createElement("label");
  label.textContent = field.name;
  label.setAttribute("for", field.name);

  let input;
  if (field.type === "select") {
    // Campo di tipo select (categorico)
    input = document.createElement("select");
    input.id = field.name;
    input.name = field.name;

    // Prima opzione vuota "-- Seleziona --"
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "-- Seleziona --";
    empty.disabled = true;
    empty.selected = true;
    input.appendChild(empty);

    // Aggiunge le opzioni disponibili dallo schema
    (field.options || []).forEach(opt => {
      const o = document.createElement("option");
      o.value = opt;
      o.textContent = opt;
      input.appendChild(o);
    });

  } else {
    // Campo numerico (default)
    input = document.createElement("input");
    input.type = "number";
    input.step = "any"; // consente decimali
    input.placeholder = "Inserisci numero";
    input.id = field.name;
    input.name = field.name;
  }

  // Assembla nel DOM
  wrap.appendChild(label);
  wrap.appendChild(input);
  return wrap;
}


// ======================================================
// 🧱 Funzione: carica dallo schema del backend la struttura dei campi
// ======================================================
async function loadSchema() {
  const r = await fetch(API_SCHEMA);
  const schema = await r.json();
  const fields = schema.fields || [];

  // Pulisce il form e aggiunge i nuovi campi dinamicamente
  formEl.innerHTML = "";
  fields.forEach(f => formEl.appendChild(createField(f)));
}


// ======================================================
// 📥 Funzione: raccoglie i dati compilati dall’utente nel form
// ======================================================
function collectFormData() {
  const inputs = formEl.querySelectorAll("input,select");
  const features = {};

  inputs.forEach(el => {
    if (!el.name) return;
    let val = el.value;
    if (val === "") return; // ignora campi vuoti

    // Prova a convertire in numero se è un input numerico
    if (el.tagName === "INPUT" && el.type === "number") {
      const num = Number(val);
      features[el.name] = Number.isFinite(num) ? num : val;
    } else {
      features[el.name] = val;
    }
  });
  return features;
}


// ======================================================
// 📊 Funzione: mostra a schermo il risultato o un errore
// ======================================================
function showResult(ok, payload) {
  resultEl.classList.remove("hidden", "error");
  if (!ok) {
    resultEl.classList.add("error");
    resultEl.innerHTML = `<strong>Errore:</strong> ${payload}`;
    return;
  }

  const { prediction, proba, used_features, message, model_name } = payload;

  // 👇 Al posto di "Predizione: <numero>", mostriamo il modello
  let html = `<strong>Modello:</strong> ${model_name || "Sconosciuto"} <br/>`;

  // Se vuoi, puoi ancora mostrare anche la predizione reale in piccolo:
  // html += `<small style="color:#a7b0d3">Predizione: ${prediction}</small><br/>`;

  if (proba) {
    html += `<strong>Probabilità:</strong> `;
    html += Object.entries(proba)
      .map(([cls, p]) => `${cls}: ${(p * 100).toFixed(2)}%`)
      .join(" | ");
    html += "<br/>";
  }
  if (used_features && used_features.length) {
    html += `<small style="color:#a7b0d3">Feature usate: ${used_features.join(", ")}</small><br/>`;
  }
  html += `<small style="color:#a7b0d3">${message}</small>`;
  resultEl.innerHTML = html;
}


// ======================================================
// 🧠 Evento: invia i dati al backend per ottenere la predizione
// ======================================================
predictBtn.addEventListener("click", async () => {
  const features = collectFormData();

  try {
    const r = await fetch(API_PREDICT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }) // 👈 chiave obbligatoria per FastAPI
    });

    if (!r.ok) {
      const err = await r.json();
      showResult(false, err.detail || "Richiesta non valida");
      return;
    }

    const data = await r.json();
    showResult(true, data);

  } catch (e) {
    showResult(false, e.message || String(e));
  }
});


// ======================================================
// 🔄 Evento: resetta i campi e nasconde il risultato
// ======================================================
resetBtn.addEventListener("click", () => {
  formEl.reset();
  resultEl.classList.add("hidden");
});


// ======================================================
// 🎲 Utility: genera numeri casuali
// ======================================================
function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randFloat(min, max, decimals = 1) {
  const n = Math.random() * (max - min) + min;
  return Number(n.toFixed(decimals));
}


// ======================================================
// 🎯 Funzione: genera valori casuali ma sensati per ciascun campo
// ======================================================
function guessRandomForField(name, type, options = []) {
  const key = name.toLowerCase();

  // Se è una select, scegli una voce a caso
  if (type === "select" && Array.isArray(options) && options.length > 0) {
    return options[randInt(0, options.length - 1)];
  }

  // Alcune euristiche per generare valori realistici
  if (key.includes("age"))                  return randInt(12, 20);      // età scolastica
  if (key.includes("bmi"))                  return randFloat(16, 30, 1);
  if (key.includes("hours") || key.includes("per_week"))
                                            return randInt(0, 14);       // ore di attività
  if (key.includes("rate") || key.includes("percent"))
                                            return randInt(60, 100);     // percentuali
  if (key.includes("level"))                return randInt(1, 5);
  if (key.includes("final_grade") || key.includes("previous") || key.endsWith("grade"))
                                            return randInt(60, 100);
  if (key.includes("participation"))        return randInt(1, 5);
  if (key.includes("motivation"))           return randInt(1, 5);
  if (key.includes("knowledge") || key.includes("skills"))
                                            return randInt(40, 100);
  if (key.includes("score"))                return randInt(40, 100);
  if (key.includes("improvement"))          return randInt(0, 30);

  // fallback generico
  return randInt(0, 100);
}


// ======================================================
// 🧩 Funzione: riempie tutti i campi automaticamente
// ======================================================
async function fillRandomFields() {
  // Recupera lo schema attuale (serve per capire i tipi e opzioni)
  const r = await fetch("/api/schema");
  if (!r.ok) throw new Error("Impossibile recuperare lo schema");
  const schema = await r.json();
  const fields = schema.fields || [];

  // Mappa nome → tipo/opzioni
  const def = {};
  fields.forEach(f => { def[f.name] = f; });

  // Itera su tutti gli input e assegna valori casuali
  const inputs = formEl.querySelectorAll("input,select");
  inputs.forEach(el => {
    const name = el.name;
    if (!name) return;

    const meta = def[name] || {};
    const type = meta.type || (el.tagName === "SELECT" ? "select" : "number");
    const options = meta.options || [];

    const value = guessRandomForField(name, type, options);

    if (el.tagName === "SELECT") {
      const found = Array.from(el.options).find(o => o.value == value || o.textContent == value);
      if (found) {
        el.value = found.value;
      } else if (el.options.length > 1) {
        el.selectedIndex = randInt(1, el.options.length - 1); // evita "-- Seleziona --"
      }
    } else {
      el.value = value;
    }
  });
}


// ======================================================
// 🎮 Evento: "Compila casuale" → riempie i campi con valori sensati
// ======================================================
fillBtn.addEventListener("click", () => {
  fillRandomFields().catch(e => {
    resultEl.classList.remove("hidden");
    resultEl.classList.add("error");
    resultEl.innerHTML = `<strong>Errore riempimento:</strong> ${e.message || e}`;
  });
});


const parseBtn = document.getElementById("parse-btn");
const textInput = document.getElementById("user-text");

// Quando l’utente clicca “Genera features dal testo”
parseBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) return alert("Scrivi una frase prima!");

  try {
    const r = await fetch("/api/parse_text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    if (!r.ok) throw new Error("Errore durante il parsing del testo.");
    const data = await r.json();
    const features = data.features || {};

    // Riempi i campi del form automaticamente
    for (const [k, v] of Object.entries(features)) {
      const el = document.getElementById(k);
      if (el) el.value = v;
    }

    resultEl.classList.remove("hidden", "error");
    resultEl.innerHTML = `<strong>✅ Features generate automaticamente!</strong><br/><small>${data.message}</small>`;

  } catch (e) {
    resultEl.classList.remove("hidden");
    resultEl.classList.add("error");
    resultEl.innerHTML = `<strong>Errore NLP:</strong> ${e.message || e}`;
  }
});


// ======================================================
// 🚀 Bootstrap iniziale: carica lo schema al caricamento della pagina
// ======================================================
loadSchema().catch(console.error);