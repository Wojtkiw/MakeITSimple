import streamlit as st
import os
import random
import tempfile
from docloader import load_file, validate_upload, scan_for_injection
from embedder_rag import create_index
from guide_generator import generate_guide, answer_question, generate_quiz

st.set_page_config(page_title="MakeITSimple", layout="wide")

# CZCIONKA
# Lexend (tekst interfejsu) + Lora (szeryfowy tytul pojecia w kartach).
FONT_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700;800&family=Lora:wght@500;600&display=swap');
html, body, .stApp, [class*="st-"], [data-testid="stMarkdownContainer"],
h1, h2, h3, h4, h5, h6, p, li, span, label, button, input, textarea, select {
    font-family: 'Lexend', system-ui, -apple-system, sans-serif;
}
h1 { font-weight: 800; letter-spacing: -0.02em; }
/* Nie nadpisuj czcionki ikon - inaczej zamiast glifu widac 'upload', 'arrow_right' itd. */
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}
</style>
"""
st.markdown(FONT_STYLE, unsafe_allow_html=True)

# TLO STRONY (zeszyt)
# Kolor tla ustawia .streamlit/config.toml osobno dla light i dark - Streamlit przelacza
# motyw po stronie przegladarki (bez rerunu Pythona), wiec to MUSI byc w configu, nie w
# st.context.theme. Tutaj dokladamy tylko linie zeszytu i czerwony margines z lewej.
PAGE_STYLE = """
<style>
.stApp {
    background-image:
        linear-gradient(90deg, transparent 40px, #5a2525 40px 42px, transparent 42px),
        repeating-linear-gradient(transparent 0 27px, #1a1a24 27px 28px) !important;
}
[data-testid="stHeader"] { background: transparent; }
/* Wieksza przerwa pod polem wgrywania, zeby material i kolejne sekcje nie kleily sie do uploadera. */
[data-testid="stFileUploader"] { margin-bottom: 2.2rem; }
</style>
"""
st.markdown(PAGE_STYLE, unsafe_allow_html=True)

st.title("MakeITSimple")
st.caption("Wgraj materiał i dostań materiały do nauki.")

# STYL KART
# Ciemna "kartka": kazda sekcja to pudelko z kolorowa kreska z lewej i lekko
# podbarwionym tlem (kolor sekcji przy alpha 0.05). Tytul pojecia szeryfowy (Lora).
CARD_STYLE = """
<style>
.karta {background:#1a1a24;border:.5px solid rgba(255,255,255,.12);border-radius:16px;
        padding:26px 30px 28px;margin:16px 0;
        background-image:repeating-linear-gradient(transparent 0 28px,rgba(255,255,255,.018) 28px 29px);}
.lbl {display:flex;align-items:center;gap:9px;font-size:12px;font-weight:500;
      letter-spacing:1.6px;text-transform:uppercase;margin:24px 0 14px;}
.lbl-top {margin-top:0;}
.lbl::before {content:'';width:4px;height:14px;border-radius:2px;background:currentColor;flex:none;}
.l-blue {color:#66ffff;} .l-teal {color:#00ff99;} .l-amber {color:#ff9933;}
.l-violet {color:#b06bff;}
.concept {font-family:'Lora',serif;font-weight:600;font-size:32px;line-height:1.15;
          color:#f6f5f1;margin:0 0 4px;}
.box {border-radius:9px;padding:14px 18px;border-left:3px solid;font-size:14px;
      line-height:1.65;margin:0;color:#d6d4ce;}
.b-teal {background:rgba(0,255,153,.05);border-left-color:#00ff99;}
.b-amber {background:rgba(255,153,51,.05);border-left-color:#ff9933;font-style:italic;}
.b-violet {background:rgba(176,107,255,.05);border-left-color:#b06bff;}
.b-violet p {margin:4px 0;font-size:13.5px;line-height:1.6;}
.b-violet b {color:#b06bff;font-weight:600;}
/* Zwijana karta: naglowek (Pojecie + nazwa) siedzi w <summary>, reszta w ciele <details>.
   Chowamy domyslny trojkat przegladarki i dajemy wlasny chevron, ktory obraca sie po rozwinieciu. */
details.karta > summary {list-style:none;cursor:pointer;position:relative;padding-right:24px;outline:none;}
details.karta > summary::-webkit-details-marker {display:none;}
details.karta > summary::after {content:'';position:absolute;right:0;top:50%;width:9px;height:9px;
        border-right:2px solid #66ffff;border-bottom:2px solid #66ffff;
        transform:translateY(-50%) rotate(45deg);transition:transform .2s ease;opacity:.85;}
details.karta[open] > summary::after {transform:translateY(-50%) rotate(225deg);}
</style>
"""

# STYL QUIZU
# Ta sama paleta co karty: ramka + tlo z niska opacity zamiast pelnych wypelnien.
# Poprawna odpowiedz zielona (#00ff99), bledny strzal usera czerwony (#ff5c5c), akcent cyan (#66ffff).
QUIZ_STYLE = """
<style>
.quiz-head {display:flex;align-items:center;gap:9px;font-size:12px;font-weight:500;
            letter-spacing:1.6px;text-transform:uppercase;color:#66ffff;margin:8px 0 4px;}
.quiz-head::before {content:'';width:4px;height:14px;border-radius:2px;background:#66ffff;flex:none;}
.qz-score {display:inline-flex;align-items:center;gap:9px;border:.5px solid rgba(102,255,255,.4);
           background:rgba(102,255,255,.07);border-radius:9px;padding:9px 16px;margin:6px 0 6px;
           color:#66ffff;font-weight:600;font-size:15px;}
.qz-q {font-size:15px;font-weight:500;color:#f6f5f1;margin:18px 0 10px;}
.qz-opt {display:flex;align-items:center;gap:10px;border-radius:9px;padding:9px 13px;margin:5px 0;
         font-size:14px;line-height:1.5;}
.qz-ok {border:.5px solid rgba(0,255,153,.45);background:rgba(0,255,153,.08);color:#eafff6;}
.qz-bad {border:.5px solid rgba(255,92,92,.45);background:rgba(255,92,92,.08);color:#ffe2e2;}
.qz-mut {color:#85837b;padding:6px 13px;}
.qz-mark {margin-left:auto;font-weight:700;}
.qz-ok .qz-mark {color:#00ff99;} .qz-bad .qz-mark {color:#ff5c5c;}
.qz-foot {font-size:12px;margin:7px 0 0;}
.qz-foot-ok {color:#00ff99;} .qz-foot-bad {color:#ff5c5c;}
</style>
"""

# RENDEROWANIE KART
# Tor strukturalny: lista kroków/części pojęcia (np. etapy algorytmu genetycznego).
def render_skladniki(items):
    if not items:
        return ""
    lista = "".join(
        f'<p><b>{i.get("nazwa", "")}</b> — {i.get("opis", "")}</p>'
        for i in items
    )
    return f'<div class="lbl l-violet">Kluczowe składniki</div><div class="box b-violet">{lista}</div>'

# Tor strukturalny: nastawy pojęcia z typową wartością (np. rozmiar populacji 50-200).
def render_parametry(items):
    if not items:
        return ""
    wiersze = []
    for i in items:
        nazwa = i.get("nazwa", "")
        opis = i.get("opis", "")
        wartosc = i.get("wartosc", "")
        opis_html = f" ({opis})" if opis else ""
        wiersze.append(f'<p><b>{nazwa}</b>{opis_html} — {wartosc}</p>')
    lista = "".join(wiersze)
    return f'<div class="lbl l-violet">Parametry</div><div class="box b-violet">{lista}</div>'

def render_card(pojecie_data):
    # Karta zwijana: <summary> to klikalny naglowek (etykieta Pojecie + nazwa) widoczny zawsze,
    # reszta (wyjasnienie, analogia, dol) odslania sie po rozwinieciu. Start zwiniety - brak atrybutu open.
    # Dol karty: skladniki (kroki/czesci) + ewentualne parametry. Oba opcjonalne -
    # pojecia czysto opisowe nie maja zadnego i karta konczy sie na analogii.
    dol_html = (render_skladniki(pojecie_data.get("skladniki", []))
                + render_parametry(pojecie_data.get("parametry", [])))
    return f"""
    <details class="karta">
        <summary>
            <div class="lbl lbl-top l-blue">Pojęcie</div>
            <div class="concept">{pojecie_data['pojecie']}</div>
        </summary>
        <div class="lbl l-teal">Wyjaśnienie pojęcia</div>
        <div class="box b-teal">{pojecie_data['wyjasnienie']}</div>
        <div class="lbl l-amber">Analogia z życia</div>
        <div class="box b-amber">{pojecie_data['analogia']}</div>
        {dol_html}
    </details>
    """

# RENDEROWANIE QUIZU (TRYB PO SPRAWDZENIU)
# Jedno pytanie po kliknieciu "Sprawdz": poprawna opcja na zielono z haczykiem, bledny strzal
# usera na czerwono z krzyzykiem, reszta wyszarzona. Litery A-D dokladamy tu - opcje z modelu sa czyste.
def render_quiz_result(q, numer, wybrana):
    poprawna = q.get("poprawna", "")
    wiersze = []
    for j, opt in enumerate(q.get("opcje", [])):
        litera = "ABCD"[j]
        if litera == poprawna:
            wiersze.append(f'<div class="qz-opt qz-ok">{litera}) {opt}<span class="qz-mark">✓</span></div>')
        elif litera == wybrana:
            wiersze.append(f'<div class="qz-opt qz-bad">{litera}) {opt}<span class="qz-mark">✗</span></div>')
        else:
            wiersze.append(f'<div class="qz-opt qz-mut">{litera}) {opt}</div>')
    pojecie = q.get("pojecie", "")
    if wybrana == poprawna:
        stopka = f'<p class="qz-foot qz-foot-ok">Twoja odpowiedź — poprawna · z karty: {pojecie}</p>'
    elif wybrana is None:
        stopka = f'<p class="qz-foot qz-foot-bad">Brak odpowiedzi · poprawna: {poprawna} · z karty: {pojecie}</p>'
    else:
        stopka = f'<p class="qz-foot qz-foot-bad">Twoja odpowiedź — błędna · poprawna: {poprawna} · z karty: {pojecie}</p>'
    return (f'<div class="qz-q">{numer}. {q.get("pytanie", "")}</div>'
            + "".join(wiersze) + stopka)

# Buduje nowa runde quizu z pojec jeszcze nieodpytanych (spoza doc["quiz_used"]).
# Gdy pula sie wyczerpie - zeruje zbior uzytych i startuje od nowa, z notka dla usera.
# Zwraca True przy sukcesie, zeby UI wiedzialo czy zrobic rerun.
def _start_quiz_round(doc):
    available = [p for p in doc["pojecia"] if p["pojecie"] not in doc["quiz_used"]]
    if not available:
        doc["quiz_used"] = set()
        doc["quiz_reset_notice"] = True
        available = list(doc["pojecia"])
    pick = random.sample(available, min(10, len(available)))
    with st.spinner("Układam pytania..."):
        try:
            questions = generate_quiz(pick)
        except RuntimeError as e:
            st.error(str(e))
            return False
    if not questions:
        st.error("Nie udało się ułożyć quizu. Spróbuj ponownie.")
        return False
    doc["quiz"] = questions
    doc["quiz_used"].update(p["pojecie"] for p in pick)
    doc["quiz_round"] += 1
    doc["quiz_checked"] = False
    doc["quiz_answers"] = None
    doc["quiz_score"] = 0
    return True

# STAN PLIKOW
# Kazdy wgrany plik trzyma wlasny stan (tekst, indeks, karty, Q&A),
# dzieki czemu przelaczanie zakladek nie miesza danych miedzy plikami.
def init_state():
    st.session_state.setdefault("docs", {})
    st.session_state.setdefault("doc_counter", 0)
    st.session_state.setdefault("last_uploaded_id", None)

def add_doc(name, text):
    st.session_state["doc_counter"] += 1
    doc_id = st.session_state["doc_counter"]
    st.session_state["docs"][doc_id] = {
        "name": name,
        "raw_text": text,
        "faiss_index": None,
        "pojecia": None,
        "qa_result": None,
        # Stan quizu ABCD - osobny per dokument, jak reszta stanu.
        "quiz": None,                # pytania biezacej rundy
        "quiz_used": set(),          # nazwy pojec juz odpytanych (anty-powtorki miedzy rundami)
        "quiz_checked": False,       # czy runda zostala sprawdzona
        "quiz_answers": None,        # wybrane litery zebrane w chwili "Sprawdz"
        "quiz_score": 0,             # wynik biezacej rundy
        "quiz_round": 0,             # nonce do kluczy radio (swieze widgety po regeneracji)
        "quiz_reset_notice": False,  # pokazac info, ze pula pojec ruszyla od nowa
    }
    return doc_id

def get_doc(doc_id):
    return st.session_state["docs"].get(doc_id)

init_state()

# WGRYWANIE PLIKU
# Nowy plik (rozpoznany po nazwie+rozmiarze) dodaje osobna zakladke i staje sie aktywny.
# Ten sam plik wgrany ponownie nie tworzy duplikatu.
uploaded_file = st.file_uploader("Wgraj plik", type=["pdf", "docx", "txt", "md"])

if uploaded_file:
    file_id = f"{uploaded_file.name}:{uploaded_file.size}"
    if file_id != st.session_state["last_uploaded_id"]:
        ok, error = validate_upload(uploaded_file.name, uploaded_file.size)
        if not ok:
            st.error(error)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.rsplit('.',1)[-1]}") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            text = load_file(tmp_path, uploaded_file.name)
            os.unlink(tmp_path)

            injected, marker = scan_for_injection(text)
            if injected:
                st.warning(f"Uwaga: wykryto podejrzany fragment w pliku ('{marker}'). "
                           f"Plik może zawierać próbę manipulacji modelem.")
                if not st.button("Tak, kontynuuj mimo to"):
                    st.stop()

            new_id = add_doc(uploaded_file.name, text)
            st.session_state["file_tabs"] = new_id
            st.session_state["last_uploaded_id"] = file_id
            st.success(f"Wgrano: {uploaded_file.name}")

# ZAKLADKI PLIKOW
# Pigulki File 1 / File 2 ... wybieraja aktywny dokument; reszta apki pracuje na nim.
doc = None
active_id = None
if st.session_state["docs"]:
    ids = list(st.session_state["docs"].keys())
    selected = st.segmented_control(
        "Pliki",
        options=ids,
        format_func=lambda i: f"File {i}",
        key="file_tabs",
        label_visibility="collapsed",
    )
    active_id = selected if selected is not None else ids[-1]
    doc = get_doc(active_id)

    st.caption(f"Aktywny plik: {doc['name']}")
    with st.expander("Podgląd tekstu (pierwsze 500 znaków)"):
        st.text(doc["raw_text"][:500])

# INDEKSOWANIE
# Indeks FAISS budowany raz na dokument (leniwie, gdy zakladka pierwszy raz aktywna).
if doc is not None and doc["faiss_index"] is None:
    if not doc["raw_text"].strip():
        st.error("Plik nie zawiera tekstu. Możliwe, że to skan (zdjęcie) lub prezentacja bez warstwy tekstowej.")
    else:
        with st.spinner("Indeksuję materiał..."):
            doc["faiss_index"] = create_index(doc["raw_text"])
        st.success("Materiał zaindeksowany — gotowy do generacji!")

# PYTANIA DO MATERIALU
# Q&A (RAG): odpowiedzi wylacznie z aktywnego dokumentu. Sekcja stoi NAD generowaniem przewodnika,
# bo RAG potrzebuje tylko indeksu (gotowy zaraz po wgraniu) - nie trzeba zjezdzac na dol po wygenerowaniu kart.
# Wynik trzymany w stanie dokumentu, zeby rerun (np. otwarcie expandera) nie kasowal odpowiedzi.
if doc is not None and doc["faiss_index"] is not None:
    st.divider()
    st.subheader("Zapytaj o materiał")
    st.caption("Odpowiedzi pochodzą wyłącznie z wgranego dokumentu (RAG). "
               "Jeśli czegoś nie ma w pliku — aplikacja to przyzna, zamiast zmyślać.")
    question = st.text_input("Twoje pytanie", key=f"qa_question_{active_id}")
    if st.button("Zapytaj") and question:
        with st.spinner("Szukam w materiale..."):
            try:
                answer, fragments = answer_question(question, doc["faiss_index"])
                doc["qa_result"] = {"answer": answer, "fragments": fragments}
            except RuntimeError as e:
                doc["qa_result"] = None
                st.error(str(e))

    if doc["qa_result"]:
        st.markdown(doc["qa_result"]["answer"])
        with st.expander("Fragmenty użyte przez RAG"):
            for i, frag in enumerate(doc["qa_result"]["fragments"], 1):
                st.markdown(f"**Fragment {i}**")
                st.text(frag.get("text", ""))

# GENEROWANIE PRZEWODNIKA
if doc is not None and doc["faiss_index"] is not None:
    if st.button("Generuj przewodnik"):
        progress_bar = st.progress(0.0)
        status = st.empty()

        def update_progress(current, total, message=""):
            progress_bar.progress(min(current / total, 1.0) if total else 0.0)
            status.text(message)

        try:
            pojecia = generate_guide(doc["raw_text"], progress_callback=update_progress)
        except RuntimeError as e:
            progress_bar.empty()
            status.empty()
            st.error(str(e))
            st.stop()

        progress_bar.empty()
        status.empty()
        if pojecia:
            doc["pojecia"] = pojecia
        else:
            st.error("Nie udało się sparsować odpowiedzi modelu. Spróbuj ponownie.")

# QUIZ ABCD
# Quiz ukladany z gotowych kart pojec. Pojawia sie dopiero po wygenerowaniu przewodnika, nad kartami.
# Reveal zbiorczy: jeden przycisk "Sprawdz" odslania wynik i poprawne odpowiedzi przy wszystkich pytaniach.
if doc is not None and doc["pojecia"]:
    st.markdown(QUIZ_STYLE, unsafe_allow_html=True)
    st.markdown('<div class="quiz-head">Sprawdź się — quiz ABCD</div>', unsafe_allow_html=True)
    if doc["quiz_reset_notice"]:
        st.info("Pokryłeś już cały materiał — pula pojęć rusza od nowa.")
        doc["quiz_reset_notice"] = False

    if doc["quiz"] is None:
        st.caption("10 pytań ABCD ułożonych z Twoich kart pojęć.")
        if st.button("Generuj quiz"):
            if _start_quiz_round(doc):
                st.rerun()

    elif not doc["quiz_checked"]:
        st.caption("Zaznacz odpowiedzi i kliknij Sprawdź.")
        for i, q in enumerate(doc["quiz"], 1):
            st.markdown(f"**{i}. {q['pytanie']}**")
            opcje_ui = [f"{'ABCD'[j]}) {opt}" for j, opt in enumerate(q["opcje"])]
            st.radio("odpowiedź", opcje_ui, index=None,
                     key=f"quiz_{active_id}_{doc['quiz_round']}_{i}",
                     label_visibility="collapsed")
        if st.button("Sprawdź"):
            # Wybory zbieramy w chwili kliku (radia jeszcze sa na ekranie); w trybie reveal ich nie ma.
            wybory = []
            score = 0
            for i, q in enumerate(doc["quiz"], 1):
                sel = st.session_state.get(f"quiz_{active_id}_{doc['quiz_round']}_{i}")
                litera = sel[0] if sel else None
                wybory.append(litera)
                if litera == q["poprawna"]:
                    score += 1
            doc["quiz_answers"] = wybory
            doc["quiz_score"] = score
            doc["quiz_checked"] = True
            st.rerun()

    else:
        n = len(doc["quiz"])
        st.markdown(f'<div class="qz-score">Wynik: {doc["quiz_score"]} / {n}</div>',
                    unsafe_allow_html=True)
        for i, q in enumerate(doc["quiz"], 1):
            wybrana = doc["quiz_answers"][i - 1] if doc["quiz_answers"] else None
            st.markdown(render_quiz_result(q, i, wybrana), unsafe_allow_html=True)
        if st.button("Wygeneruj kolejne 10"):
            if _start_quiz_round(doc):
                st.rerun()

# KARTY POJEC
if doc is not None and doc["pojecia"]:
    st.markdown(CARD_STYLE, unsafe_allow_html=True)
    st.caption(f"Wygenerowano {len(doc['pojecia'])} kart pojęć")
    for p in doc["pojecia"]:
        st.markdown(render_card(p), unsafe_allow_html=True)
