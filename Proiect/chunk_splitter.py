import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_data
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
from langchain_core.documents import Document as LangchainDocument

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

def chunk_function(
    chunk_size: int,
    knowledge_base: list[LangchainDocument],
) -> list[LangchainDocument]:
    MARKDOWN_SEPARATORS = [
        r"\n{2,}",              # paragrafe separate
        r"\bDefini(ţ|t)ia\b",   # definirea termenilor
        r"\bTeorema\b",         # teoreme
        r"[0-9]+\.[0-9]+",      # subsectiuni numerotate
        r"\n",                  # linii noi
        r" "                    # fallback
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

docs_processed = chunk_function(256, load_data())

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
fig = pd.Series(lengths).hist()
plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
plt.xlabel("Tokens per chunk")
plt.ylabel("Number of chunks")
plt.show()