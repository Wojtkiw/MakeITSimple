import json
import time
from openai import OpenAI
import streamlit as st
from embedder_rag import retrieve_docs

# Prompt systemowy wysyłany do modelu przy każdej partii tekstu.
# Nakazuje zwrócić czysty JSON z listą pojęć, analogii i terminów do wyjaśnienia.
SYSTEM_PROMPT = """Jesteś ekspertem od tłumaczenia wiedzy akademickiej na zrozumiały język.
Na podstawie WYŁĄCZNIE podanego kontekstu wygeneruj przewodnik nauki w formacie JSON.

ODBIORCA: Student po maturze — zna ogólne pojęcia, ale nie jest specjalistą w tej dziedzinie.

FORMAT ODPOWIEDZI — zwróć WYŁĄCZNIE poprawny JSON, bez markdown, bez ```json:
{{"pojecia": [
  {{
    "pojecie": "nazwa pojęcia",
    "wyjasnienie": "Max 3 zdania wyłącznie techniczne, bez analogii i przykładów z życia (te są w osobnych polach). Wyjaśnij pojęcie tak, by ktoś, kto NIE widział materiału źródłowego, w pełni je zrozumiał: podaj mechanizm działania, cel i intuicję stojącą za pojęciem. NIE przepisuj skrótowych haseł z fragmentu — rozwiń je własną wiedzą w spójne, samodzielne wyjaśnienie pełnymi zdaniami. Wyjaśnienie ma być pełniejsze niż materiał źródłowy, a nie jego powtórzeniem.",
    "analogia": "1-2 zdania. Jeden żywy obraz z życia codziennego, który oddaje istotę pojęcia i jest łatwy do zapamiętania (haczyk pamięciowy) — wzór: 'zakleszczenie = dwie osoby na wąskim moście, które nie mogą się wyminąć'. Najlepszy haczyk to jedno zdanie; drugie tylko jeśli trzeba połączyć obraz z pojęciem. NIE: abstrakcja, mgła, góry. TAK: codzienne urządzenia, sytuacje z pracy/szkoły/domu. Tylko analogia — bez tłumaczenia technicznego.",
    "skladniki": [
      {{"nazwa": "nazwa kroku, części lub elementu", "opis": "kilka słów: co to / co robi"}}
    ],
    "parametry": [
      {{"nazwa": "SAM symbol/oznaczenie parametru, jak najkrótsze — użyj standardowego symbolu z wiedzy eksperckiej (np. 'N' = rozmiar populacji, 'p_c' = prawd. krzyżowania, 's' = stride, 'lr' = learning rate, 'w' = waga, 'b' = bias), nawet jeśli materiał nie podaje go wprost. NIE wpisuj tu opisowej nazwy — znaczenie idzie do pola 'opis'. Dopiero gdy parametr naprawdę nie ma przyjętego symbolu, wpisz 1-2 słowa nazwy", "opis": "czym jest ten parametr / za co odpowiada, np. 'rozmiar populacji', 'prawdopodobieństwo krzyżowania' — MUSI różnić się od pola nazwa, nie powtarzaj tego samego tekstu", "wartosc": "opcjonalnie typowa wartość lub zakres, np. '50-200'; jeśli jej nie znasz, zostaw pusty string"}}
    ]
  }}
]}}

DÓŁ KARTY — pola "skladniki" / "parametry" (oba opcjonalne; to DWA ROZŁĄCZNE rodzaje informacji — ten sam element NIE może trafić do obu sekcji jednocześnie):
- "skladniki": NATURALNE części składowe pojęcia — albo jego BUDULEC/anatomia (np. dla drzewa decyzyjnego: korzeń, węzeł, gałąź, liść; dla sieci neuronowej: warstwa wejściowa, ukryte, wyjściowa), albo KROKI działania, gdy pojęcie jest procesem (np. dla spadku gradientowego: gradient, aktualizacja parametrów, iteracja). Dla rzeczy o strukturze wypisz jej ELEMENTY, nie sam przebieg. Wypisz je KOMPLETNIE, po kolei — gdy pojęcie ma więcej niż jeden składnik, jest ich zwykle 3-5; jeśli wypisałeś tylko jeden, prawie na pewno pomijasz resztę, więc wróć i dopisz pozostałe. W polu "opis" DODAJ informację (co to robi / po co), nie powtarzaj samej nazwy. Jeśli pojęcie jest czysto opisowe i nie ma naturalnych części — zostaw [].
- "parametry": LICZBOWE nastawy/hiperparametry, które się dobiera lub uczy (mają symbol i typową wartość, np. lr, N, p_c, w, b) — NIE myl ich ze składnikami; jeśli coś wpisałeś już w "skladniki", nie powtarzaj go tutaj. Jeśli pojęcie ma takie nastawy, wypisz każdą jako PARĘ: krótki symbol w polu "nazwa" (np. "N", "p_c", "k") + wyjaśnienie czym jest w polu "opis" (np. "rozmiar populacji", "prawdopodobieństwo krzyżowania"). NIGDY nie wpisuj tego samego tekstu w "nazwa" i "opis" — to ma być symbol + jego znaczenie, jak "N" = "rozmiar populacji". Pole "wartosc" jest opcjonalne (typowa liczba/zakres albo pusty string). Jeśli pojęcie nie ma sensownych parametrów — zostaw [].
- Wiele pojęć (zwłaszcza z innych dziedzin) nie ma ani składników, ani parametrów — wtedy zostaw oba puste ([]). Karta skończy się wtedy na analogii. To jest poprawne i lepsze niż sztuczne wypełnianie pól.

ZASADY:
- Karty twórz WYŁĄCZNIE dla pojęć, które fragment nazywa lub definiuje — nie dodawaj kart dla pojęć, których fragment nie wymienia.
- Samą TREŚĆ wyjaśnień rozwijaj swoją wiedzą ekspercką, żeby były zrozumiałe i pełniejsze niż źródło. Trzymaj się tematu danego pojęcia — nie zmieniaj tematu ani nie wprowadzaj nowych, osobnych pojęć zamiast wyjaśnienia.
- Wypisz WSZYSTKIE pojęcia zdefiniowane w tym fragmencie — każdy termin, który fragment definiuje, dostaje osobną kartę. NIE streszczaj do kilku najważniejszych, NIE pomijaj żadnego.
- Opisuj każde pojęcie niezależnie. NIE wymyślaj powiązań między pojęciami, których nie ma w kontekście (np. "wagi są związane z PCA" jeśli kontekst tego nie mówi).

Kontekst z materiałów użytkownika:
{context}"""

# Model Groq (otwarty model, wymog zaliczenia). Na obrone: scout (swiezy budzet 500K/dzien, NIE jest
# modelem rozumujacym wiec zwraca czysty JSON) bo dzienny limit llama-3.3-70b-versatile (100K) wyczerpany,
# a gpt-oss-120b jako model rozumujacy gubil JSON na pelnych plikach (preambula rozumowania + uciecie).
# Powrot na 70B / proba 120B: odkomentuj odpowiednia linie.
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# MODEL = "openai/gpt-oss-120b"
# MODEL = "llama-3.3-70b-versatile"

# Stałe sterujące podziałem tekstu i limitem tokenów Groq.
BATCH_SIZE = 1800                       # max znaków tekstu w jednej porcji (mniej pojęć = pełniejsze, nieucięte karty)
MAX_OUTPUT_TOKENS = 4500                # sufit długości odpowiedzi modelu (z zapasem na bogate wyjaśnienia)
TPM_LIMIT = 28000                       # limit tokenów na minutę (scout = 30K TPM, margines bezpieczeństwa)
TPM_REFILL_PER_SEC = TPM_LIMIT / 60     # ile tokenów budżetu wraca na sekundę
NEXT_BATCH_COST = 5800                  # szacowany koszt jednej porcji (wejście ~1300 + max_tokens 4500)
MAX_RATE_LIMIT_WAITS = 6                # ile razy max odczekujemy na odnowienie limitu, potem błąd
RATE_LIMIT_WAIT_SECONDS = 60            # ile czekać gdy Groq nie poda retry-after
MAX_REASONABLE_WAIT = 90                # dłuższe czekanie = limit dzienny, nie minutowy


# Wyjątek rzucany gdy Groq odrzuci zapytanie z powodu limitu tokenów.
# Przechowuje retry_after z nagłówka odpowiedzi, żeby wiedzieć ile czekać.
class _RateLimit(Exception):
    def __init__(self, retry_after=None):
        super().__init__()
        self.retry_after = retry_after


# Bezpieczna konwersja wartości nagłówka HTTP na liczbę; zwraca None jeśli się nie uda.
def _parse_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Model bywa zwraca klucze JSON po polsku z ogonkami ("pojęcia" zamiast "pojecia").
# Reszta kodu oczekuje kluczy ASCII, więc sprowadzamy SAME KLUCZE (nie wartości) do wersji bez ogonków.
_KEY_DIACRITICS = str.maketrans("ąćęłńóśżźĄĆĘŁŃÓŚŻŹ", "acelnoszzACELNOSZZ")


def _canon_keys(obj):
    if isinstance(obj, dict):
        return {k.translate(_KEY_DIACRITICS): _canon_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_canon_keys(x) for x in obj]
    return obj


# Normalizacja tekstu do porównań: małe litery, bez białych znaków na brzegach.
def _normalize(s):
    return s.strip().lower()


# Wyciąga "rdzeń" nazwy pojęcia, obcinając doprecyzowania w nawiasach, np. "SVM (klasyfikator)" -> "svm".
# Używane żeby nie traktować "SVM" i "SVM (kernel)" jako dwóch różnych kart.
def _concept_core(name):
    for sep in ("[", "("):
        if sep in name:
            name = name.split(sep)[0]
    return _normalize(name)


# Usuwa duplikaty pojęć po scaleniu wyników z wielu partii.
# Porównuje po rdzeniu nazwy, więc "ReLU" i "relu" to ten sam wpis.
def _dedupe(pojecia):
    seen = set()
    out = []
    for p in pojecia:
        core = _concept_core(p.get("pojecie", ""))
        if core and core not in seen:
            seen.add(core)
            out.append(p)
    return out


# Dzieli tekst na partie o maksymalnej długości batch_size znaków, nie rwąc linii w połowie.
# Mniejsze partie = tańsze zapytania = rzadziej uderzamy w limit tokenów na minutę.
def _split_into_batches(text, batch_size=BATCH_SIZE):
    lines = text.split("\n")
    batches = []
    current = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > batch_size and current:
            batches.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        batches.append("\n".join(current))
    return batches


# Wysyła jedną partię tekstu do modelu i zwraca listę pojęć oraz liczbę pozostałych tokenów z nagłówka.
# Używamy with_raw_response żeby odczytać nagłówki limitów przed sparsowaniem odpowiedzi.
def _generate_batch(client, batch_text):
    prompt = SYSTEM_PROMPT.format(context=batch_text)
    try:
        raw_response = client.chat.completions.with_raw_response.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # ekstrakcja pojec, nie kreatywne pisanie: 0 daje powtarzalna liczbe kart i zblizone wyniki miedzy formatami
            max_tokens=MAX_OUTPUT_TOKENS
        )
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "too large" in msg or "tokens per minute" in msg or "tpm" in msg:
            retry_after = None
            response = getattr(e, "response", None)
            if response is not None:
                retry_after = _parse_number(response.headers.get("retry-after"))
            raise _RateLimit(retry_after)
        if "timeout" in msg:
            raise RuntimeError("Groq nie odpowiedział w czasie. Spróbuj ponownie.")
        raise RuntimeError(f"Błąd API: {e}")

    remaining_tokens = _parse_number(raw_response.headers.get("x-ratelimit-remaining-tokens"))
    response = raw_response.parse()
    raw = response.choices[0].message.content.strip()
    # Llama 4 Scout często owija JSON w bloki markdown mimo zakazu w prompcie
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if "```" in raw:
            raw = raw[:raw.rfind("```")]
        raw = raw.strip()
    try:
        data = _canon_keys(json.loads(raw))
        return data["pojecia"], remaining_tokens
    except (json.JSONDecodeError, KeyError):
        return [], remaining_tokens


# Odlicza sekundy w pasku postępu zamiast zamrażać UI na time.sleep.
# Dzięki temu użytkownik widzi żywe odliczanie a nie zamrożony ekran.
def _countdown(seconds, current, total, progress_callback):
    seconds = max(1, int(round(seconds)))
    if not progress_callback:
        time.sleep(seconds)
        return
    for left in range(seconds, 0, -1):
        progress_callback(current, total, f"Partia {current}/{total} gotowa — przygotowuję kolejną ({left} s)...")
        time.sleep(1)


# Główna funkcja: dzieli tekst na partie, wysyła każdą do modelu i scala wyniki.
# Śledzi czas żeby pokazać ETA, i proaktywnie czeka gdy budżet minutowy jest bliski wyczerpania.
def generate_guide(raw_text, progress_callback=None):
    client = OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url=st.secrets["GROQ_BASE_URL"]
    )
    batches = _split_into_batches(raw_text)
    total = len(batches)
    all_pojecia = []
    waits = 0
    i = 0
    start_time = time.time()

    while i < total:
        eta = ""
        if i > 0:
            remaining_est = int((time.time() - start_time) / i * (total - i))
            eta = f" — pozostało ~{remaining_est} s"
        if progress_callback:
            progress_callback(i, total, f"Generuję partię {i + 1}/{total}{eta}...")
        try:
            pojecia, remaining_tokens = _generate_batch(client, batches[i])
        except _RateLimit as e:
            wait_for = e.retry_after if e.retry_after else RATE_LIMIT_WAIT_SECONDS
            if wait_for > MAX_REASONABLE_WAIT:
                minutes = int(wait_for // 60) + 1
                raise RuntimeError(
                    "Wyczerpał się dzienny limit darmowego Groqa. "
                    f"Reset za ~{minutes} min — spróbuj ponownie później."
                )
            waits += 1
            if waits > MAX_RATE_LIMIT_WAITS:
                raise RuntimeError(
                    "Groq ciągle odrzuca zapytania (limit tokenów). "
                    "Spróbuj ponownie za kilka minut albo wgraj mniejszy plik."
                )
            _countdown(wait_for, i, total, progress_callback)
            continue

        all_pojecia.extend(pojecia)
        i += 1

        # Proaktywne czekanie: jeśli budżet minutowy nie starczy na kolejną
        # partię, odczekaj dokładnie tyle, ile potrzeba na odnowienie tokenów.
        if i < total and remaining_tokens is not None and remaining_tokens < NEXT_BATCH_COST:
            needed = NEXT_BATCH_COST - remaining_tokens
            wait_for = min(needed / TPM_REFILL_PER_SEC + 1, RATE_LIMIT_WAIT_SECONDS)
            _countdown(wait_for, i, total, progress_callback)

    if progress_callback:
        progress_callback(total, total, "Składam przewodnik...")

    if not all_pojecia:
        return None
    return _dedupe(all_pojecia)


# Q&A z ugruntowaniem w dokumencie (RAG, wersja ścisła).
# Pobiera k fragmentów najbliższych znaczeniowo pytaniu i każe modelowi odpowiedzieć
# WYŁĄCZNIE z nich; brak pokrycia -> jawne "Nie znalazłem tego w dokumencie".
# Zwraca (odpowiedz, uzyte_fragmenty), żeby UI mogło pokazać dowód RAG.
# k=8, bo embedder all-MiniLM jest anglojęzyczny i na polskim daje płaskie odległości
# (właściwy fragment bywa dopiero ~6. w rankingu); węższe k gubił trafny chunk i model
# odpowiadał "Nie znalazłem", choć temat był w materiale.
def answer_question(question, faiss_index, k=8, client=None):
    fragments = retrieve_docs(question, faiss_index, k=k)
    context = "\n\n".join(f.get("text", "") for f in fragments)
    if client is None:
        client = OpenAI(
            api_key=st.secrets["GROQ_API_KEY"],
            base_url=st.secrets["GROQ_BASE_URL"]
        )
    system = (
        "Odpowiadasz na pytania WYŁĄCZNIE na podstawie poniższego kontekstu z materiału użytkownika. "
        "Jeśli odpowiedzi nie ma w kontekście, napisz dokładnie: «Nie znalazłem tego w dokumencie». "
        "Nie korzystaj z wiedzy spoza kontekstu i nie zgaduj.\n\n"
        f"KONTEKST:\n{context}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question}
            ],
            temperature=0.2,
            max_tokens=600
        )
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "tokens per minute" in msg or "tpm" in msg:
            raise RuntimeError("Przekroczony limit tokenów Groq. Spróbuj ponownie za chwilę.")
        if "timeout" in msg:
            raise RuntimeError("Groq nie odpowiedział w czasie. Spróbuj ponownie.")
        raise RuntimeError(f"Błąd API: {e}")
    answer = response.choices[0].message.content.strip()
    return answer, fragments


# Przepuszcza tylko pytania w pełnym formacie: 4 opcje i poprawna litera A-D.
# Model czasem urwie opcję albo poda zły indeks — wadliwe wpisy odsiewamy, żeby UI
# nie wywaliło się na braku pola przy renderowaniu odpowiedzi.
def _valid_question(q):
    if not isinstance(q, dict):
        return False
    opcje = q.get("opcje")
    if not isinstance(opcje, list) or len(opcje) != 4:
        return False
    if str(q.get("poprawna", "")).strip().upper() not in {"A", "B", "C", "D"}:
        return False
    return bool(str(q.get("pytanie", "")).strip())


# Quiz ABCD układany z gotowych kart pojęć.
# Dostaje JUŻ wybraną listę kart (losowanie i pilnowanie pokrycia materiału robi UI),
# dla każdej karty prosi o jedno pytanie z czterema odpowiedziami i jedną poprawną.
# Jedno tanie wywołanie na rundę — bez batchowania, bezpieczne dla limitu w dniu obrony.
# Zwraca listę pytań [{pojecie, pytanie, opcje[4], poprawna}] z odsianymi wadliwymi wpisami.
def generate_quiz(concept_cards, client=None):
    if not concept_cards:
        return []
    if client is None:
        client = OpenAI(
            api_key=st.secrets["GROQ_API_KEY"],
            base_url=st.secrets["GROQ_BASE_URL"]
        )
    # Każde pojęcie podajemy nazwą + wyjaśnieniem, żeby pytanie trzymało się materiału, a nie wiedzy ogólnej modelu.
    pojecia_blok = "\n\n".join(
        f'POJĘCIE: {c.get("pojecie", "")}\nWYJAŚNIENIE: {c.get("wyjasnienie", "")}'
        for c in concept_cards
    )
    prompt = (
        "Układasz quiz ABCD dla studenta na podstawie podanych pojęć. "
        "Dla KAŻDEGO pojęcia ułóż dokładnie jedno pytanie sprawdzające ZROZUMIENIE (nie samą definicję), "
        "z czterema odpowiedziami, z których dokładnie jedna jest poprawna, a trzy to sensowne ale błędne "
        "dystraktory. Pisz po polsku, zwięźle.\n\n"
        "Zwróć WYŁĄCZNIE poprawny JSON, bez markdown, bez ```:\n"
        '{"pytania": [{"pojecie": "nazwa pojęcia", "pytanie": "treść pytania", '
        '"opcje": ["tekst A", "tekst B", "tekst C", "tekst D"], "poprawna": "A"}]}\n'
        'Pole "opcje" to czysty tekst odpowiedzi BEZ prefiksów typu "A)". '
        'Pole "poprawna" to litera A, B, C lub D wskazująca pozycję poprawnej odpowiedzi na liście "opcje".\n\n'
        f"POJĘCIA:\n{pojecia_blok}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2200
        )
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "tokens per minute" in msg or "tpm" in msg:
            raise RuntimeError("Przekroczony limit tokenów Groq. Spróbuj ponownie za chwilę.")
        if "timeout" in msg:
            raise RuntimeError("Groq nie odpowiedział w czasie. Spróbuj ponownie.")
        raise RuntimeError(f"Błąd API: {e}")

    raw = response.choices[0].message.content.strip()
    # Scout potrafi owinąć JSON w blok markdown mimo zakazu — zdejmujemy fencing jak w _generate_batch.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if "```" in raw:
            raw = raw[:raw.rfind("```")]
        raw = raw.strip()
    try:
        data = _canon_keys(json.loads(raw))
        pytania = data["pytania"]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise RuntimeError("Nie udało się ułożyć quizu. Spróbuj ponownie.")

    quiz = []
    for q in pytania:
        if _valid_question(q):
            q["poprawna"] = str(q["poprawna"]).strip().upper()
            quiz.append(q)
    return quiz
