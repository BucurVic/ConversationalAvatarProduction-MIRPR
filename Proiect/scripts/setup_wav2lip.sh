#!/usr/bin/env bash

echo "=============================="
echo "     Wav2Lip SETUP SCRIPT"
echo "=============================="

TARGET_DIR="external/Wav2Lip"

echo "-> Creăm directoarele necesare..."
mkdir -p external

# 1. Clonăm repo-ul Wav2Lip
if [ ! -d "$TARGET_DIR/.git" ]; then
    echo "-> Clonăm Wav2Lip în external/Wav2Lip..."
    git clone https://github.com/Rudrabha/Wav2Lip.git $TARGET_DIR
else
    echo "-> Wav2Lip este deja clonat. Continuăm."
fi

# 2. Instalăm dependințele necesare
echo ""
echo "-> Instalăm dependințele..."

pip install numpy opencv-python moviepy librosa tqdm

# 3. Descărcăm modelul pre-antrenat
echo ""
echo "-> Descărcăm modelul pre-antrenat Wav2Lip..."

cd $TARGET_DIR

mkdir -p checkpoints

curl -L -o checkpoints/wav2lip_gan.pth \
"https://huggingface.co/spaces/SmilingWolf/wav2lip/resolve/main/Wav2Lip/checkpoints/wav2lip_gan.pth?download=true"

echo "=============================================="
echo "  Wav2Lip instalat cu succes!"
echo "  Găsit în: external/Wav2Lip/"
echo "=============================================="