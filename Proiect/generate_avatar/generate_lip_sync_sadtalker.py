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

    project_root = Path(__file__).resolve().parents[1]  # adjust as needed
    python310 = project_root / "venv310" / "bin" / "python"

    if not python310.exists():
        raise RuntimeError(f"Python 3.10 venv not found at {python310}")

    # cmd = [
    #     str(python310),
    #     "inference.py",
    #     "--driven_audio", str(audio_path),
    #     "--source_image", str(image_path),
    #     "--result_dir", str(output_dir),
    #     "--still"
    # ]

    # cmd = [
    # str(python310),
    # "inference.py",
    # "--driven_audio", str(audio_path),
    # "--source_image", str(image_path),
    # "--result_dir", str(output_dir),
    # "--preprocess", "full",
    # "--enhancer", "gfpgan",
    # "--batch_size", "2",
    # "--size", "256"
    # ]

    cmd = [
    str(python310),
    "inference.py",
    "--driven_audio", str(audio_path),
    "--source_image", str(image_path),
    "--save_dir", str(output_dir),
    "--preprocess", "crop",
    "--enhancer", "gfpgan",
    "--batch_size", "2",
    "--size", "256",
    "--expression_scale", "1.0",
    "--background_enhancer", "gfpgan"
    ]



    env = os.environ.copy()
    env["PATH"] = f"{python310.parent}:" + env.get("PATH", "")

    subprocess.run(cmd, cwd=sadtalker_repo, env=env, check=True)

    # SadTalker generează un .mp4 în folderul de output
    videos = sorted(output_dir.glob("*.mp4"), key=os.path.getmtime)
    if not videos:
        raise RuntimeError("SadTalker nu a generat niciun video.")
    return str(videos[-1])