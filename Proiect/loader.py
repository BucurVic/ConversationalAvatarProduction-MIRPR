import fitz
import re
from tqdm import tqdm
from langchain_core.documents import Document as LangchainDocument

def eliminate_enter(text: str):
    return text.replace('\n', ' ').replace('  ', ' ')

def load_pdf_to_documents(pdf_path: str):
    pdf = fitz.open(pdf_path)
    text_all = ""
    for page in pdf:
        text_all += page.get_text("text") + "\n"

    pdf.close()

    text_all = eliminate_enter(text_all)

    pattern = re.compile(r'(CAPITOLUL\s+\d+\s+[^\n]+)')

    parts = pattern.split(text_all)

    dataSet = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i+1].strip() if i + 1 < len(parts) else ""
        dataSet.append({
            "text": f"Title: {title}: {content}",
            "source": title
        })

    return [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in tqdm(dataSet, desc="Loading PDF chapters")
    ]

def load_data():
    vector_files = ["./data/manual2022.pdf"]
    ds = []
    for path in vector_files:
        ds.extend(load_pdf_to_documents(path))
    return ds