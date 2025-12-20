import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from generate_lip_sync_wav2lip import wav2lip_generate_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path la imagine (input_img.png)")
    parser.add_argument("--audio", required=True, help="Path la audio (answer.wav)")
    parser.add_argument("--output", required=True, help="Folder output")
    parser.add_argument("--repo", required=False, help="Path la Wav2Lip") 
    
    args = parser.parse_args()

    wav2lip_generate_video(
        image_path=args.image,
        audio_path=args.audio,
        output_dir=args.output,
        wav2lip_repo=args.repo
    )