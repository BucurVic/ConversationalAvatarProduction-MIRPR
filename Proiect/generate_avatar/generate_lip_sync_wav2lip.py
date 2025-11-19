import subprocess
from pathlib import Path
import cv2


def create_static_video(image_path, output_video, duration=4, fps=25):
    image_path = Path(image_path)
    output_video = Path(output_video)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    for _ in range(int(duration * fps)):
        out.write(frame)

    out.release()


def wav2lip_generate_video(
    image_path: str,
    audio_path: str,
    output_dir: str,
    wav2lip_repo: str
) -> str:

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    static_video = output_dir / "static_avatar.mp4"
    output_video = output_dir / "avatar_lipsync.mp4"

    # 1️⃣ Generează video static
    create_static_video(image_path, static_video)

    # 2️⃣ Construim path-uri ABSOLUTE pentru Wav2Lip
    static_video_abs = str(static_video.resolve())
    output_video_abs = str(output_video.resolve())
    audio_path_abs = str(Path(audio_path).resolve())
    checkpoint_path_abs = str(Path(wav2lip_repo) / "checkpoints" / "wav2lip_gan.pth")

    # 3️⃣ Comandă Wav2Lip cu path-uri ABSOLUTE
    cmd = [
        "python",
        "inference.py",
        "--face", static_video_abs,
        "--audio", audio_path_abs,
        "--outfile", output_video_abs,
        "--checkpoint_path", checkpoint_path_abs,
    ]

    subprocess.run(cmd, cwd=wav2lip_repo, check=True)

    return output_video_abs