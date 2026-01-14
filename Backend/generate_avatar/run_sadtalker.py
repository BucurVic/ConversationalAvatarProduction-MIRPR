import argparse
import sys
from pathlib import Path

# --- SETUP IMPORTURI ---
# Adăugăm folderul curent la path pentru a putea importa modulul 'generate_avatar'
# (Presupunând că run_sadtalker.py este în rădăcina proiectului sau în folderul scripts)
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Importăm funcția robustă pe care am creat-o anterior
# Aceasta conține deja logica pentru Windows, PYTHONPATH și argumentele pentru video 3D
try:
    from generate_avatar.generate_lip_sync_sadtalker import sadtalker_generate_video
except ImportError:
    # Fallback: Dacă scriptul e rulat direct din folderul generate_avatar
    sys.path.append(str(current_dir.parent))
    from generate_avatar.generate_lip_sync_sadtalker import sadtalker_generate_video

def main():
    parser = argparse.ArgumentParser(description="Script de rulare manuală SadTalker")
    parser.add_argument("--image", required=True, help="Calea către imaginea sursă")
    parser.add_argument("--audio", required=True, help="Calea către fișierul audio (wav/mp3)")
    parser.add_argument("--output", required=True, help="Folderul unde se salvează rezultatul")
    parser.add_argument("--repo", required=False, help="Calea către folderul SadTalker (external/SadTalker)")
    
    args = parser.parse_args()

    # Calea default către repo dacă nu este dată
    repo_path = args.repo
    if not repo_path:
        # Încercăm să ghicim calea relativ la acest script
        potential_path = Path(__file__).resolve().parent.parent / "external" / "SadTalker"
        if potential_path.exists():
            repo_path = str(potential_path)

    print("--- Start SadTalker Manual Run ---")
    try:
        video_path = sadtalker_generate_video(
            image_path=args.image,
            audio_path=args.audio,
            output_dir=args.output,
            sadtalker_repo=repo_path
        )
        print(f"\n✅ SUCCESS! Video generat la:\n{video_path}")
    except Exception as e:
        print(f"\n❌ EROARE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()