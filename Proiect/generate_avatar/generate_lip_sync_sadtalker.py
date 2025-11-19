import subprocess
from pathlib import Path
import os

def sadtalker_generate_video(
    image_path: str,
    audio_path: str,
    output_dir: str,
    sadtalker_repo: str
) -> str:

    image_path = Path(image_path).resolve()
    audio_path = Path(audio_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "inference.py",
        "--driven_audio", str(audio_path),
        "--source_image", str(image_path),
        "--result_dir", str(output_dir),
        "--still"
    ]

    subprocess.run(cmd, cwd=sadtalker_repo, check=True)

    # SadTalker generează un .mp4 în folderul de output
    videos = sorted(output_dir.glob("*.mp4"), key=os.path.getmtime)
    if not videos:
        raise RuntimeError("SadTalker nu a generat niciun video.")
    return str(videos[-1])