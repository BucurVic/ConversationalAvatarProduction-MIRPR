import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    source_image = Path(args.image).resolve()
    audio = Path(args.audio).resolve()
    output_dir = Path(args.output).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,         # python din venv310
        "inference.py",
        "--driven_audio", str(audio),
        "--source_image", str(source_image),
        "--result_dir", str(output_dir),
        "--still",
    ]

    print("[run_sadtalker] Rulez comanda:")
    print(" ", " ".join(cmd))
    print("[run_sadtalker] cwd:", repo)

    subprocess.run(cmd, cwd=str(repo), check=True)


if __name__ == "__main__":
    main()