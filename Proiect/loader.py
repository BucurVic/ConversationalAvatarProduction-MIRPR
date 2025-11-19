import fitz
import re
from tqdm import tqdm
from langchain_core.documents import Document as LangchainDocument
from unidecode import unidecode

def clean_text_block(text: str):
    """
    Curăță un bloc de text (elimină diacritice și spații inutile),
    dar DOAR după ce am separat capitolele.
    """
    # 1. Elimină diacriticele
    text = unidecode(text)
    # 2. Înlocuiește newline cu spațiu și reduce spațiile multiple
    text = text.replace('\n', ' ').replace('  ', ' ')
    return text.strip()

def load_pdf_to_documents(pdf_path: str):
    print(f"Se încarcă PDF-ul: {pdf_path}")
    pdf = fitz.open(pdf_path)
    text_all = ""
    for page in pdf:
        text_all += page.get_text("text") + "\n"
    pdf.close()

    # --- AICI ERA PROBLEMA: NU curățăm textul înainte de split! ---
    # Păstrăm textul original (cu \n) ca să putem găsi titlurile.

    # Regex ajustat: Caută "CAPITOLUL X" la început de rând sau după un newline
    # Group 1: Titlul complet (ex: "CAPITOLUL 1 Vectori...")
    # (?m) activează modul multiline
    pattern = re.compile(r'(CAPITOLUL\s+\d+.*)', re.IGNORECASE)

    parts = pattern.split(text_all)

    dataSet = []
    
    # Dacă prima parte (înainte de primul capitol) are text, o tratăm ca "Introducere"
    if parts[0].strip():
         dataSet.append({
            "text": clean_text_block(parts[0]),
            "source": "Introducere / Prefata"
        })

    # Iterăm prin părțile găsite (titlu -> conținut -> titlu -> conținut)
    # parts[1] este primul titlu, parts[2] este conținutul primului capitol, etc.
    for i in range(1, len(parts), 2):
        raw_title = parts[i].strip()
        raw_content = parts[i+1].strip() if i + 1 < len(parts) else ""
        
        # Curățăm ACUM titlul și conținutul, separat
        clean_title = clean_text_block(raw_title)
        clean_content = clean_text_block(raw_content)
        
        # Combinăm titlul cu conținutul pentru a nu pierde informația din titlu în vector
        full_text = f"{clean_title}: {clean_content}"

        dataSet.append({
            "text": full_text,
            "source": clean_title # Sursa este doar titlul curat
        })

    print(f"S-au identificat {len(dataSet)} secțiuni/capitole.")
    
    return [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in tqdm(dataSet, desc="Procesare Capitole")
    ]

def load_data():
    # Asigură-te că calea este corectă
    vector_files = ["./data/manual2022.pdf"] 
    ds = []
    for path in vector_files:
        ds.extend(load_pdf_to_documents(path))
    return ds

