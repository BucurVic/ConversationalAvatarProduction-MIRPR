# Face2Learn - Asistent Educațional Multimodal

[cite_start]**Face2Learn** este un sistem inteligent bazat pe Inteligență Artificială care combină modele de limbaj mari (LLM), sinteză vocală (TTS) și animație facială pentru a crea o experiență de învățare naturală și interactivă[cite: 10].

## Descriere
[cite_start]Scopul proiectului este de a transforma informațiile academice în explicații conversaționale, vizuale și auditive, crescând astfel implicarea, accesibilitatea și retenția informației[cite: 11]. [cite_start]Sistemul procesează întrebări în limbaj natural și generează un videoclip cu un avatar animat (lip-sync) care oferă explicația didactică[cite: 26, 49].

## Arhitectură și Tehnologii
[cite_start]Proiectul utilizează o arhitectură Full-Stack modernă[cite: 12, 244]:

* [cite_start]**Backend:** REST API dezvoltat cu **FastAPI**, optimizat pentru latență scăzută și rulare multi-platformă (CUDA/Metal)[cite: 13]. [cite_start]Utilizează tehnici avansate de RAG (Hybrid Search, Cosine Similarity)[cite: 13, 243].
* [cite_start]**Frontend:** Interfață grafică intuitivă dezvoltată în **React** și **Vite**, cu design modern (Glassmorphism)[cite: 14].
* **AI & Optimizare(QLoRA):**
    * [cite_start]LLM: Modele optimizate (ex. Llama 3) folosind QLORA[cite: 243].
    * [cite_start]Sinteză Video: Wav2Lip High-Res / SadTalker pentru sincronizare speech->video[cite: 243].
    * [cite_start]Reranking: Arhitectură Two-Stage Retrieval pentru precizia răspunsurilor[cite: 153].
    * [cite_start]**QLoRA (Quantized Low-Rank Adaptation):** S-a utilizat această tehnică pentru a adapta modelul generativ la domeniul educațional românesc și pentru a reduce necesarul de memorie (încărcare în 4-biți).


**Membri:**
* [cite_start]Bucur Victor Sever [cite: 6]
* [cite_start]Popoviciu Luca [cite: 6]
* [cite_start]Porcar Cezar [cite: 6]
* [cite_start]Potra-Rațiu Darius [cite: 6]
* [cite_start]Preduca Matei [cite: 6]