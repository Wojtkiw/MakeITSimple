import streamlit as st
import os
import tempfile
from docloader import load_file, validate_upload, scan_for_injection
from embedder_rag import create_index
from guide_generator import generate_guide, answer_question

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

# KARTY POJEC
if doc is not None and doc["pojecia"]:
    st.markdown(CARD_STYLE, unsafe_allow_html=True)
    st.caption(f"Wygenerowano {len(doc['pojecia'])} kart pojęć")
    for p in doc["pojecia"]:
        st.markdown(render_card(p), unsafe_allow_html=True)

# PYTANIA DO MATERIALU
# Q&A (RAG): odpowiedzi wylacznie z aktywnego dokumentu. Wynik trzymany w stanie tego dokumentu,
# zeby otwarcie expandera (rerun) nie kasowalo odpowiedzi i nie mieszalo jej miedzy zakladkami.
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
