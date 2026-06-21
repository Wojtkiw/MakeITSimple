import os
import fitz
import docx

INJECTION_MARKERS = [
    "ignore previous instructions",
    "forget your instructions",
    "you are now",
    "act as",
    "ignore all previous",
    "new instructions:",
    "system prompt:",
]

def scan_for_injection(text):
    text_lower = text.lower()
    for marker in INJECTION_MARKERS:
        if marker in text_lower:
            return True, marker
    return False, None

def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def load_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_file(file_path, filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    loaders = {"pdf": load_pdf, "docx": load_docx, "txt": load_txt, "md": load_md}
    if ext not in loaders:
        raise ValueError(f"Nieobsługiwany typ pliku: .{ext}")
    return loaders[ext](file_path)

def validate_upload(filename, size_bytes):
    allowed = {"pdf", "docx", "txt", "md"}
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return False, f"Nieobsługiwany typ pliku. Dozwolone: {', '.join(allowed)}"
    if size_bytes > 10 * 1024 * 1024:
        return False, "Plik za duży. Maksymalny rozmiar: 10MB"
    return True, None
