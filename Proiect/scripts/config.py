from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# LM Studio
LM_STUDIO_URL = "http://localhost:1234/v1"
LLM_MODEL = "mistral-7b-instruct-v0.3.Q4_K_M.gguf"

# Vectorstore
DB_FAISS_PATH = str(BASE_DIR / "vectorstore")
EMBEDDING_MODEL_NAME = "thenlper/gte-small"

# SadTalker localizat în Proiect/external/SadTalker
SADTALKER_REPO = str(BASE_DIR / "external" / "SadTalker")
WAV2LIP_REPO = str(BASE_DIR / "external" / "Wav2Lip")

# Avatar input image
USER_IMAGE = str(BASE_DIR / "generate_avatar" / "input_img.png")
