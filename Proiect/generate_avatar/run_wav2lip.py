import argparse
from generate_lip_sync_wav2lip import wav2lip_generate_video

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--audio", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--repo", required=True)
args = parser.parse_args()

wav2lip_generate_video(
    image_path=args.image,
    audio_path=args.audio,
    output_dir=args.output,
    wav2lip_repo=args.repo
)