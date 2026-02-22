#!/usr/bin/env python3
"""
Prepare an Italian "common life" instruction dataset.

Outputs:
  - datasets/life_common_it.jsonl
  - datasets/life_common_eval_it.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


CATALOG: List[Dict[str, object]] = [
    {
        "category": "home_admin",
        "prompts": [
            "Come organizzo le bollette senza dimenticarle?",
            "Metodo semplice per gestire spese di casa ogni mese?",
            "Come dividere le spese domestiche in famiglia?",
        ],
        "answers": [
            "Usa un sistema fisso in 4 passi: 1) elenco bollette con scadenza, 2) addebito automatico dove possibile, 3) promemoria 7 giorni prima, 4) controllo finale a fine mese. Tieni un foglio unico con importo, data, stato pagato.",
            "Fai un budget 50/30/20 adattato: 50 percento spese essenziali, 30 percento variabili, 20 percento risparmio o debito. Rivedi una volta a settimana e correggi subito gli extra.",
            "Definite regole chiare: spese fisse in quota proporzionale al reddito, spese variabili con fondo comune. Registrate tutto nello stesso file per evitare discussioni a fine mese.",
        ],
        "must_have": ["scadenza", "promemoria", "foglio"],
    },
    {
        "category": "job_search",
        "prompts": [
            "Come trovo lavoro in modo pratico in 30 giorni?",
            "Strategia concreta per candidature efficaci?",
            "Come migliorare cv e colloquio velocemente?",
        ],
        "answers": [
            "Piano 30 giorni: giorni 1-3 CV mirato, giorni 4-10 20 candidature personalizzate, giorni 11-20 follow up, giorni 21-30 simulazioni colloquio. Ogni candidatura deve avere parole chiave della job description.",
            "Usa la regola 70/20/10: 70 percento candidature mirate, 20 percento networking, 10 percento candidature esplorative. Traccia azienda, ruolo, data, esito e follow up.",
            "CV in una pagina con risultati misurabili, non solo mansioni. Per il colloquio prepara 5 esempi STAR: situazione, compito, azione, risultato.",
        ],
        "must_have": ["candidature", "follow up", "cv"],
    },
    {
        "category": "study",
        "prompts": [
            "Come studiare meglio con poco tempo?",
            "Metodo per non dimenticare quello che studio?",
            "Come preparare un esame in modo ordinato?",
        ],
        "answers": [
            "Studia in blocchi da 25 minuti con 5 minuti pausa. Dopo ogni blocco fai un mini ripasso attivo a voce alta senza guardare appunti.",
            "Usa ripetizione dilazionata: ripasso dopo 1 giorno, 3 giorni, 7 giorni, 14 giorni. Focalizzati su domande e risposte, non su rilettura passiva.",
            "Dividi il programma in moduli settimanali con obiettivi chiari. Ogni domenica fai una verifica con quiz o simulazione.",
        ],
        "must_have": ["ripasso", "blocchi", "obiettivi"],
    },
    {
        "category": "digital_safety",
        "prompts": [
            "Come proteggere account e dati personali?",
            "Regole base per non cadere in phishing?",
            "Cosa fare se sospetto un furto account?",
        ],
        "answers": [
            "Attiva password manager, password uniche e autenticazione a due fattori. Aggiorna sistema e app in automatico.",
            "Non cliccare link urgenti senza verifica. Controlla dominio reale, errori nel testo e richieste di dati sensibili.",
            "Cambia subito password, disconnetti sessioni attive, attiva 2FA e contatta assistenza ufficiale. Se c e danno economico valuta denuncia.",
        ],
        "must_have": ["due fattori", "password", "verifica"],
    },
    {
        "category": "health_non_clinical",
        "prompts": [
            "Come migliorare energia durante il giorno?",
            "Routine base per sonno migliore?",
            "Come costruire abitudini sane senza stress?",
        ],
        "answers": [
            "Base pratica: orario sonno regolare, idratazione, movimento leggero quotidiano e pasti semplici. Riduci caffeina nel tardo pomeriggio.",
            "Per dormire meglio: stessa ora di sonno e risveglio, luce bassa la sera, niente schermo negli ultimi 30 minuti, camera fresca e buia.",
            "Parti piccolo: una sola abitudine per 2 settimane. Misura su calendario e non cercare perfezione, cerca continuita.",
        ],
        "must_have": ["sonno", "idratazione", "routine"],
    },
    {
        "category": "public_services",
        "prompts": [
            "Come gestire documenti e scadenze burocratiche?",
            "Metodo semplice per pratiche amministrative?",
            "Come non perdere documenti importanti?",
        ],
        "answers": [
            "Crea archivio digitale e cartaceo con 5 cartelle: identita, casa, lavoro, salute, auto. Aggiungi promemoria annuali per rinnovi.",
            "Per ogni pratica usa checklist: documento richiesto, modulo, costo, ufficio, data invio, ricevuta. Conserva ricevuta sempre.",
            "Scannerizza subito i documenti ricevuti, rinomina con data e tipo, salva in cloud e su backup locale.",
        ],
        "must_have": ["checklist", "ricevuta", "scadenze"],
    },
    {
        "category": "mobility",
        "prompts": [
            "Come risparmiare su trasporti ogni mese?",
            "Come organizzare spostamenti casa lavoro?",
            "Cosa controllare prima di un viaggio breve?",
        ],
        "answers": [
            "Confronta abbonamento, carnet e pay per ride. Se fai tratte fisse valuta abbonamento, se variabile usa monitoraggio spesa settimanale.",
            "Prepara due alternative di percorso e un margine di 15 minuti. Riduci ritardi con orari stabili e controllo traffico la sera prima.",
            "Checklist viaggio: documenti, ricariche, orari, meteo, piano B. Tieni numeri utili in note offline.",
        ],
        "must_have": ["abbonamento", "checklist", "piano b"],
    },
    {
        "category": "family_relations",
        "prompts": [
            "Come ridurre conflitti in casa per le faccende?",
            "Come parlare di soldi in coppia senza litigare?",
            "Come organizzare i compiti familiari?",
        ],
        "answers": [
            "Accordo semplice: lista compiti, responsabile, frequenza, controllo settimanale di 10 minuti. Focus su regole, non su colpe.",
            "Parlate di budget in una riunione fissa e breve, con numeri davanti. Definite limite spesa personale e obiettivi comuni.",
            "Usa calendario condiviso e rotazione dei compiti. Se un compito salta, spostalo subito con nuova data.",
        ],
        "must_have": ["regole", "budget", "calendario"],
    },
    {
        "category": "emergency_basics",
        "prompts": [
            "Cosa fare in emergenza domestica base?",
            "Come preparare un piano emergenze in famiglia?",
            "Numeri e azioni da ricordare in urgenza?",
        ],
        "answers": [
            "Prepara kit base, numeri utili e punto di ritrovo. In emergenza metti prima in sicurezza le persone e chiama i soccorsi se serve.",
            "Definite ruoli: chi chiama aiuto, chi prende documenti, chi assiste bambini o anziani. Fate una prova ogni 6 mesi.",
            "Memorizza 112 e indirizzo completo di casa. In caso medico grave chiama subito 112 e segui istruzioni operatore.",
        ],
        "must_have": ["112", "sicurezza", "piano"],
    },
    {
        "category": "personal_finance",
        "prompts": [
            "Come creare un fondo emergenza da zero?",
            "Come uscire da spese impulsive?",
            "Strategia pratica per ridurre debiti piccoli?",
        ],
        "answers": [
            "Obiettivo iniziale: 1000 euro, poi 3 mesi di spese essenziali. Automatica un trasferimento il giorno dello stipendio.",
            "Applica regola 24 ore per acquisti non urgenti e limite mensile per categoria. Se superi il limite blocca nuova spesa.",
            "Ordina debiti per tasso o importo. Paga minimo su tutti e extra sul primo target fino a chiusura.",
        ],
        "must_have": ["fondo", "spese", "debiti"],
    },
    {
        "category": "time_management",
        "prompts": [
            "Come gestire tempo tra lavoro e vita privata?",
            "Metodo rapido per priorita giornaliere?",
            "Come evitare procrastinazione pratica?",
        ],
        "answers": [
            "Usa 3 priorita massime al giorno. Blocca calendario per attivita profonde e lascia slot per imprevisti.",
            "Classifica compiti in urgente importante, importante non urgente, delegabile, eliminabile. Inizia dal compito ad alto impatto.",
            "Riduci attrito: prepara ambiente la sera prima, inizia con 10 minuti e usa timer. Parti piccolo ma subito.",
        ],
        "must_have": ["priorita", "calendario", "timer"],
    },
    {
        "category": "consumer_rights",
        "prompts": [
            "Come gestire un acquisto difettoso in modo corretto?",
            "Passi per un reclamo efficace?",
            "Come raccogliere prove prima di contestazione?",
        ],
        "answers": [
            "Conserva ricevuta, foto del problema e comunicazioni. Contatta venditore per iscritto con richiesta chiara e data.",
            "Reclamo efficace: descrizione fatti, prova allegata, richiesta specifica, scadenza risposta. Usa canale tracciabile.",
            "Prima della contestazione raccogli cronologia eventi, prove di pagamento e riferimenti ordine. Mantieni tono neutro.",
        ],
        "must_have": ["ricevuta", "reclamo", "prove"],
    },
]


SAFETY_APPENDS = [
    "Per situazioni mediche, legali o fiscali complesse, consulta un professionista qualificato.",
    "Se c e un rischio immediato per la sicurezza, chiama il 112.",
]


def _sample_entry(rng: random.Random, category: Dict[str, object]) -> Dict[str, object]:
    prompt = rng.choice(category["prompts"])  # type: ignore[index]
    answer = rng.choice(category["answers"])  # type: ignore[index]
    if rng.random() < 0.2:
        answer = f"{answer} {rng.choice(SAFETY_APPENDS)}"
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"category": category["category"]},
    }


def build_train_dataset(target_size: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    per_category = max(20, target_size // max(1, len(CATALOG)))

    for cat in CATALOG:
        for _ in range(per_category):
            rows.append(_sample_entry(rng, cat))

    while len(rows) < target_size:
        rows.append(_sample_entry(rng, rng.choice(CATALOG)))

    rng.shuffle(rows)
    return rows[:target_size]


def build_eval_dataset(size: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed + 991)
    rows: List[Dict[str, object]] = []

    for cat in CATALOG:
        prompt = rng.choice(cat["prompts"])  # type: ignore[index]
        must_have = cat["must_have"]  # type: ignore[index]
        risk = "high" if cat["category"] in {"emergency_basics"} else "normal"
        rows.append(
            {
                "prompt": prompt,
                "category": cat["category"],
                "must_have": must_have,
                "should_have": ["passi", "chiaro", "pratico"],
                "risk": risk,
            }
        )

    while len(rows) < size:
        cat = rng.choice(CATALOG)
        rows.append(
            {
                "prompt": rng.choice(cat["prompts"]),  # type: ignore[index]
                "category": cat["category"],
                "must_have": cat["must_have"],
                "should_have": ["passi", "pratico"],
                "risk": "normal",
            }
        )

    rng.shuffle(rows)
    return rows[:size]


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build life common IT datasets.")
    parser.add_argument(
        "--train-output",
        default="datasets/life_common_it.jsonl",
        help="Output JSONL for SFT/RAG corpus.",
    )
    parser.add_argument(
        "--eval-output",
        default="datasets/life_common_eval_it.jsonl",
        help="Output JSONL for evaluation prompts.",
    )
    parser.add_argument("--train-size", type=int, default=2400, help="Training row count.")
    parser.add_argument("--eval-size", type=int, default=240, help="Evaluation row count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = build_train_dataset(target_size=args.train_size, seed=args.seed)
    eval_rows = build_eval_dataset(size=args.eval_size, seed=args.seed)

    train_path = Path(args.train_output)
    eval_path = Path(args.eval_output)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    print(f"Train dataset: {train_path} ({len(train_rows)} rows)")
    print(f"Eval dataset:  {eval_path} ({len(eval_rows)} rows)")


if __name__ == "__main__":
    main()
