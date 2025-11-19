#!/usr/bin/env bash

echo "=============================="
echo "   SadTalker SETUP SCRIPT"
echo "=============================="

# Folder în care vom instala SadTalker
TARGET_DIR="external/SadTalker"

echo "-> Creăm directoarele necesare..."
mkdir -p external
mkdir -p $TARGET_DIR

# 1. Clonăm SadTalker
if [ ! -d "$TARGET_DIR/.git" ]; then
    echo "-> Clonăm SadTalker în external/SadTalker..."
    git clone https://github.com/OpenTalker/SadTalker.git $TARGET_DIR
else
    echo "-> SadTalker este deja clonat, trecem mai departe."
fi

# 2. Instalăm dependințele
echo ""
echo "-> Instalăm dependințe în mediul virtual curent..."

pip install -r $TARGET_DIR/requirements.txt

# Instalăm explicit pachete care lipsesc în unele setup-uri
pip install imageio imageio-ffmpeg

# 3. Descărcăm modelele SadTalker
echo ""
echo "-> Descărcăm modelele SadTalker..."

cd $TARGET_DIR

# Scriptul lor de download
if [ -f "scripts/download_models.sh" ]; then
    chmod +x scripts/download_models.sh
    ./scripts/download_models.sh
else
    echo "[EROARE] Scriptul de download modele nu există!"
    echo "Repo-ul SadTalker s-a schimbat."
    echo "Verificați structura repository-ului."
    exit 1
fi

cd ../../

echo "=============================================="
echo "  SadTalker instalat cu succes!"
echo "  Găsit în: external/SadTalker/"
echo "=============================================="
