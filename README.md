# Face2Learn - Asistent Educațional Multimodal

**Face2Learn** este un sistem inteligent bazat pe Inteligență Artificială care combină modele de limbaj mari (LLM), sinteză vocală (TTS) și animație facială pentru a crea o experiență de învățare naturală și interactivă.

## Descriere
Scopul proiectului este de a transforma informațiile academice în explicații conversaționale, vizuale și auditive, crescând astfel implicarea, accesibilitatea și retenția informațieiSistemul procesează întrebări în limbaj natural și generează un videoclip cu un avatar animat (lip-sync) care oferă explicația didactică.

## Arhitectură și Tehnologii
Proiectul utilizează o arhitectură Full-Stack modernă:

* **Backend:** REST API dezvoltat cu **FastAPI**, optimizat pentru latență scăzută și rulare multi-platformă (CUDA/Metal)Utilizează tehnici avansate de RAG (Hybrid Search, Cosine Similarity)
* **Frontend:** Interfață grafică intuitivă dezvoltată în **React** și **Vite**, cu design modern (Glassmorphism)
* **AI & Optimizare(QLoRA):**
    * LLM: Modele optimizate (ex. Llama 3) folosind QLORA
    * Sinteză Video: Wav2Lip High-Res / SadTalker pentru sincronizare speech->video
    * Reranking: Arhitectură Two-Stage Retrieval pentru precizia răspunsurilor
    * **QLoRA (Quantized Low-Rank Adaptation):** S-a utilizat această tehnică pentru a adapta modelul generativ la domeniul educațional românesc și pentru a reduce necesarul de memorie (încărcare în 4-biți).


**Membri:**
* Bucur Victor Sever 
* Popoviciu Luca 
* Porcar Cezar 
* Potra-Rațiu Darius 
* Preduca Matei 