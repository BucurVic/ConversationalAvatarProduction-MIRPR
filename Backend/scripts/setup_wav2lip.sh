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

# # 2. Instalăm dependințele necesare
# echo ""
# echo "-> Instalăm dependințele..."


# 3. Descărcăm modelul pre-antrenat
echo ""
echo "-> Descărcăm modelul pre-antrenat Wav2Lip..."

cd $TARGET_DIR

mkdir -p checkpoints
