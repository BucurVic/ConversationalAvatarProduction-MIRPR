FILEID="15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk"
FILENAME="wav2lip_gan.pth"

cd external/Wav2Lip/checkpoints

wget --no-check-certificate \
     "https://docs.google.com/uc?export=download&id=${FILEID}" \
     -O ${FILENAME}
