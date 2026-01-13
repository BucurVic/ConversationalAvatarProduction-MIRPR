from pathlib import Path

# Calculăm rădăcina proiectului (Proiect/)
# __file__ este scripts/config.py -> parent e scripts/ -> parent.parent e Proiect/
BASE_DIR = Path(__file__).resolve().parent.parent

# --- DATABASE ---
# Folderul unde ai creat indexul FAISS (vectorstoretmp conform discuției anterioare)
DB_FAISS_PATH = str(BASE_DIR / "workflow" / "vectorstoretmp")

# --- MODELE AI ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Calea către modelul Llama GGUF
# Se presupune că e în: Proiect/models/Meta-Llama...
# LLM_MODEL_PATH = str(BASE_DIR / "models" / "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")
LLM_MODEL_PATH = str(BASE_DIR / "models" / "llama-3-8b-instruct.Q4_K_M.gguf")

# --- AVATAR & VIDEO ---
# Repo-ul Wav2Lip (clonat în external)
WAV2LIP_REPO = str(BASE_DIR / "external" / "Wav2Lip")

# Imaginea de bază pentru avatar
USER_IMAGE = str(BASE_DIR / "generate_avatar" / "input_img.png")

# Folderul unde se vor salva videoclipurile generate
VIDEO_OUTPUT_DIR = BASE_DIR / "workflow" / "runtime" / "avatar_wav2lip" / "video"