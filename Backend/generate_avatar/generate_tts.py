import subprocess
import os
import platform
from pathlib import Path

# --- CONFIGURARE ---
CURRENT_DIR = Path(__file__).resolve().parent
PIPER_MODEL = CURRENT_DIR / "ro_RO-mihai-medium.onnx"

def tts_piper(text, output_file="answer.wav"):
    # Verificăm modelul (necesar pe ambele sisteme)
    if not PIPER_MODEL.exists():
        raise FileNotFoundError(f"[TTS] Modelul nu există la: {PIPER_MODEL}")

    output_file_str = str(Path(output_file).resolve())
    system_os = platform.system()

    # =========================================================================
    # CAZUL 1: WINDOWS (Folosim fix-ul cu executabil local și fișier binar)
    # =========================================================================
    if system_os == "Windows":
        # Căutăm executabilul în folderul 'piper' de lângă script
        piper_binary = CURRENT_DIR / "piper" / "piper.exe"
        
        if not piper_binary.exists():
            raise FileNotFoundError(
                f"[TTS WINDOWS] Nu găsesc 'piper.exe' la: {piper_binary}\n"
                "Pe Windows, asigură-te că ai copiat folderul 'piper' descărcat lângă acest script."
            )

        # Pregătim textul și fișierul temporar
        clean_text = text.replace("\n", " ").replace("\r", "").strip()
        temp_input_file = CURRENT_DIR / "temp_tts_input.txt"
        
        try:
            # Scriem textul UTF-8 pe disc
            with open(temp_input_file, "w", encoding="utf-8") as f:
                f.write(clean_text)

            cmd = [
                str(piper_binary),
                "--model", str(PIPER_MODEL),
                "--output_file", output_file_str
            ]
            
            # Trimitem fișierul ca input BINAR (fix pentru diacritice Windows)
            with open(temp_input_file, "rb") as audio_pipe:
                subprocess.run(
                    cmd, 
                    stdin=audio_pipe, 
                    check=True,
                    cwd=str(piper_binary.parent) # Ajută la găsirea DLL-urilor
                )
                
        except subprocess.CalledProcessError as e:
            print(f"[TTS Error] Windows Piper a eșuat: {e}")
            raise e
        finally:
            if temp_input_file.exists():
                try: os.remove(temp_input_file)
                except: pass

    # =========================================================================
    # CAZUL 2: MAC / LINUX (Metoda originală, simplă)
    # =========================================================================
    else:
        # Pe Mac presupunem că 'piper' este instalat global (în PATH)
        # sau poți pune calea absolută dacă îl ai într-un loc specific
        cmd = [
            "piper", 
            "--model", str(PIPER_MODEL),
            "--output_file", output_file_str
        ]

        try:
            # Pe Mac, pipe-ul UTF-8 funcționează corect direct din memorie
            subprocess.run(
                cmd, 
                input=text.encode("utf-8"), 
                check=True
            )
        except FileNotFoundError:
            print("[TTS Error] Comanda 'piper' nu a fost găsită în PATH.")
            print("Pe Mac, asigură-te că ai instalat piper (ex: brew sau binary în /usr/local/bin).")
            raise
        except subprocess.CalledProcessError as e:
            print(f"[TTS Error] Mac Piper a eșuat: {e}")
            raise e

    return output_file