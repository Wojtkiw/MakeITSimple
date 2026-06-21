import os
import fitz
import docx

# Frazy typowe dla prompt injection (po angielsku, bo tak je formułuje atakujący w treści pliku).
# Materiał użytkownika to dane, nie polecenia - jeśli któraś fraza się pojawi, ostrzegamy zamiast wykonywać.
INJECTION_MARKERS = [
    "ignore previous instructions",
    "forget your instructions",
    "you are now",
    "act as",
    "ignore all previous",
    "new instructions:",
    "system prompt:",
]

# Skanuje materiał pod kątem prób przejęcia instrukcji modelu; zwraca pierwszą trafioną frazę.
def scan_for_injection(text):
    text_lower = text.lower()
    for marker in INJECTION_MARKERS:
        if marker in text_lower:
            return True, marker
    return False, None

# Wyciąga tekst ze wszystkich stron PDF-a (PyMuPDF/fitz).
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Czyta tekst z akapitów DOCX, pomijając puste linie.
def load_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# Wczytuje zwykły plik tekstowy jako UTF-8.
def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Markdown traktujemy jak czysty tekst - formatowanie nie przeszkadza modelowi.
def load_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Dobiera loader po rozszerzeniu pliku; nieznany typ kończy się jawnym błędem.
def load_file(file_path, filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    loaders = {"pdf": load_pdf, "docx": load_docx, "txt": load_txt, "md": load_md}
    if ext not in loaders:
        raise ValueError(f"Nieobsługiwany typ pliku: .{ext}")
    return loaders[ext](file_path)

# Sprawdza typ i rozmiar przed wczytaniem - odrzuca nieobsługiwane formaty i pliki ponad 10 MB.
def validate_upload(filename, size_bytes):
    allowed = {"pdf", "docx", "txt", "md"}
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return False, f"Nieobsługiwany typ pliku. Dozwolone: {', '.join(allowed)}"
    if size_bytes > 10 * 1024 * 1024:
        return False, "Plik za duży. Maksymalny rozmiar: 10MB"
    return True, None
