import json
import re
from llama_cpp import Llama

# --- CONFIGURARE ---
LLM_MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
OUTPUT_FILE = "dataset_finetuning_ready.jsonl"

# Datele tale brute (le-am pus aici pentru simplitate, dar le poți încărca din fișier)
raw_data = [
  
  {
    "intrebare": "Ce este un segment orientat?",
    "raspuns_asteptat": "Un segment de dreapta pentru care s-a precizat care dintre capetele sale este originea si care extremitatea, se numeste segment orientat."
  },
  {
    "intrebare": "Cum se definește egalitatea a două segmente orientate?",
    "raspuns_asteptat": "Spunem ca doua segmente orientate AB si CD sunt egale daca A=C si B=D, cu alte cuvinte, daca ele au aceeasi origine si aceeasi extremitate."
  },
  {
    "intrebare": "Ce este un vector liber?",
    "raspuns_asteptat": "Se numeste vector liber o clasa de echivalenta de segmente orientate, in raport cu relatia de echipolenta."
  },
  {
    "intrebare": "Cum se definește adunarea vectorilor folosind regula triunghiului?",
    "raspuns_asteptat": "Vectorul OB se numeste suma vectorilor a si b si se noteaza cu a+b."
  },
  {
    "intrebare": "Cum este definit produsul unui vector cu un scalar lambda?",
    "raspuns_asteptat": "Produsul vectorului a cu scalarul este, prin definitie, un vector, notat la caracterizat in modul urmator: (i) modulul lui da este dat de ||lambda a|| := |lambda| * ||a||, (ii) directia lui da coincide cu directia lui a; (iii) sensul lui da coincide cu sensul lui a daca lambda>0 sau cu sensul opus sensului lui a daca lambda<0."
  },
  {
    "intrebare": "Ce înseamnă că niște vectori sunt liniar dependenți?",
    "raspuns_asteptat": "Vectorii a_1, a_2, ..., a_k se numesc liniar dependenti daca exista numerele reale lambda_1, ..., lambda_k nu toate nule, astfel incat lambda_1*a_1 + lambda_2*a_2 + ... + lambda_k*a_k = 0."
  },
  {
    "intrebare": "Când sunt doi vectori liniar dependenți, din punct de vedere geometric?",
    "raspuns_asteptat": "Doi vectori sunt liniar dependenti daca si numai daca sunt coliniari."
  },
  {
    "intrebare": "Când sunt trei vectori liniar dependenți, din punct de vedere geometric?",
    "raspuns_asteptat": "Pentru ca trei vectori sa fie liniar dependenti este necesar si suficient ca ei sa fie coplanari."
  },
  {
    "intrebare": "Care este condiția de coliniaritate a trei puncte A, B, C?",
    "raspuns_asteptat": "Trei puncte A, B, C sunt coliniare daca si numai daca intre vectorii lor de pozitie exista o relatie de forma alpha*r_A + beta*r_B + gamma*r_C = 0, in care coeficientii verifica relatia alpha + beta + gamma = 0."
  },
  {
    "intrebare": "Care este condiția de coplanaritate a patru puncte A, B, C, D?",
    "raspuns_asteptat": "Patru puncte sunt A, B, C, D sunt coplanare daca si numai daca intre vectorii lor de poizitie exista o relatie de forma alpha*r_A + beta*r_B + gamma*r_C + delta*r_D = 0, in care coeficientii nu sunt toti nuli si verifica relatia alpha + beta + gamma + delta = 0."
  },
  {
    "intrebare": "Care este vectorul de poziție al centrului de greutate G al unui triunghi?",
    "raspuns_asteptat": "vectorul de pozitie al punctului G de intersectie a medianelor (centrul de greutate): r = (1/3)*(r_1 + r_2 + r_3)."
  },
  {
    "intrebare": "Care este vectorul de poziție al centrului cercului înscris (I) într-un triunghi?",
    "raspuns_asteptat": "(I)r = 1/(a+b+c) * (a*r_1 + b*r_2 + c*r_3)."
  },
  {
    "intrebare": "Care este condiția din Teorema lui Ceva?",
    "raspuns_asteptat": "Atunci cevienele AM, BN si CP sunt concurente daca si numai daca lambda*mu*nu = -1."
  },
  {
    "intrebare": "Care este condiția din Teorema lui Menelaus?",
    "raspuns_asteptat": "Pentru ca in membrul drept al ecuatiei (1.12.64) sa avem vectorul nul, trebuie sa punem lambda*mu*nu = 1."
  },
  {
    "intrebare": "Cum se definește produsul scalar a doi vectori?",
    "raspuns_asteptat": "Se numeste produs scalar al celor doi vectori numarul real, notat ab, egal cu produsul dintre normele celor doi vectori si al cosinusului unghiului dintre ei, adica: a*b = ||a||*||b||*cos phi"
  },
  {
    "intrebare": "Care este expresia produsului scalar în coordonate?",
    "raspuns_asteptat": "produsul scalar a doi vectori, dati prin componentele lor relativ la un sistem de coordonate rectangular Oxyz, se exprima prin formula a*b = a_1*b_1 + a_2*b_2 + a_3*b_3."
  },
  {
    "intrebare": "Când sunt doi vectori perpendiculari (ortogonali)?",
    "raspuns_asteptat": "Doi vectori a si b sunt perpendiculari daca si numai daca produsul lor scalar este egal cu zero: a*b = 0."
  },
  {
    "intrebare": "Cum se definește produsul vectorial?",
    "raspuns_asteptat": "Produsul vectorial dintre vectorul a si vectorul b este, prin definitie, vectorul, notat prin a x b, determinat prin urmatoarele conditii: 1) daca vectorii a si b sunt coliniari, atunci... a x b este egal cu zero. 2) daca cei doi vectori nu sunt coliniari... (i) lungimea vectorului a x b este egala cu ||a||*||b||*|sin phi| (ii) vectorul a x b este perpendicular pe ambii vectori a si b; (iii) tripletul de vectori (a, b, a x b) este direct."
  },
  {
    "intrebare": "Care este expresia produsului vectorial în coordonate?",
    "raspuns_asteptat": "a x b = (a_2*b_3 - a_3*b_2)i + (a_3*b_1 - a_1*b_3)j + (a_1*b_2 - a_2*b_1)k."
  },
  {
    "intrebare": "Cum se definește produsul mixt a trei vectori?",
    "raspuns_asteptat": "Fie a, b si c trei vectori. Se numeste produs mixt al celor trei vectori numarul (a,b,c) := (a x b) * c."
  },
  {
    "intrebare": "Care este expresia produsului mixt în coordonate?",
    "raspuns_asteptat": "(a,b,c) = det([[a_1, a_2, a_3], [b_1, b_2, b_3], [c_1, c_2, c_3]])."
  },
  {
    "intrebare": "Când sunt trei vectori coplanari, folosind produsul mixt?",
    "raspuns_asteptat": "Pentru ca trei vectori a, b si c sa fie coplanari este necesar si suficient ca produsul lor mixt sa fie egal cu zero: (a,b,c) = 0."
  },
  {
    "intrebare": "Care este ecuația generală a dreptei în plan?",
    "raspuns_asteptat": "Se numeste ecuatie de gradul intai sau ecuatie liniara relativ la necunoscutele x si y o ecuatie de forma Ax + By + C = 0; unde A; B; C 2 R, iar coeficientii A si B nu se anuleaza simultan."
  },
  {
    "intrebare": "Care este ecuația dreptei prin tăieturi?",
    "raspuns_asteptat": "ecuatia va deveni x/a + y/b = 1: (2.2.7) Ecuatia (2.2.7) se numeste ecuatia dreptei prin taieturi."
  },
  {
    "intrebare": "Care este ecuația vectorială a dreptei în plan?",
    "raspuns_asteptat": "r = r0 + ta; (2.3.2) se numeste ecuatia vectoriala a dreptei."
  },
  {
    "intrebare": "Care este ecuația canonică a dreptei în plan?",
    "raspuns_asteptat": "sistemul (2.3.3) este echivalent cu ecuatia (x - x0) / l = (y - y0) / m ; (2.3.4) care se numeste ecuatia canonica a dreptei in plan."
  },
  {
    "intrebare": "Care este condiția de perpendicularitate a două drepte date prin ecuațiile generale?",
    "raspuns_asteptat": "conditia necesara si suficienta pentru ca dreptele (2.7.1) si (2.7.2) sa fie perpendiculare: A1*A2 + B1*B2 = 0."
  },
  {
    "intrebare": "Care este formula distanței de la un punct la o dreaptă?",
    "raspuns_asteptat": "formulele pentru abaterea si distanta de la un punct M0(x0, y0) pana la dreapta (2.6.4) se pot scrie d = |Ax0 + By0 + C| / sqrt(A^2 + B^2)."
  },
  {
    "intrebare": "Care este ecuația vectorială a planului?",
    "raspuns_asteptat": "r = r0 + sv + tw; (3.1.2) ecuatie care se numeste ecuatia vectoriala a planului."
  },
  {
    "intrebare": "Care este ecuația generală a planului?",
    "raspuns_asteptat": "Se numeste ecuatie liniara (generala) relativ la necunoscutele x; y; z o ecuatie de forma Ax + By + Cz + D = 0; unde cel putin unul dintre coeficientii A; B; C ai necunoscutelor este diferit de zero."
  },
  {
    "intrebare": "Care este ecuația planului prin tăieturi?",
    "raspuns_asteptat": "x/a + y/b + z/c - 1 = 0: (3.1.13) Ecuatia (3.1.13) se numeste ecuatia planului prin taieturi."
  },
  {
    "intrebare": "Care este formula distanței de la un punct la un plan?",
    "raspuns_asteptat": "d = |A*x0 + B*y0 + C*z0 + D| / sqrt(A^2 + B^2 + C^2)."
  },
  {
    "intrebare": "Care sunt ecuațiile canonice ale unei drepte în spațiu?",
    "raspuns_asteptat": "Ecuatiile (3.2.4) se numesc ecuatiile canonice ale dreptei... x - x0 / l = y - y0 / m = z - z0 / n."
  },
  {
    "intrebare": "Care este condiția de perpendicularitate a două plane?",
    "raspuns_asteptat": "planele sunt perpendiculare daca si numai daca A1*A2 + B1*B2 + C1*C2 = 0."
  },
  {
    "intrebare": "Care este condiția de paralelism a două plane?",
    "raspuns_asteptat": "planele sunt paralele exact atunci cand cei doi vectori normali sunt paraleli, adica daca si numai daca A1/A2 = B1/B2 = C1/C2."
  },
  {
    "intrebare": "Care este formula distanței de la un punct la o dreaptă în spațiu?",
    "raspuns_asteptat": "formula d = ||(r1 - r0) x a|| / ||a|| ne da distanta de la punctul M1, de vector de pozitie r1 la dreapta."
  },
  {
    "intrebare": "Care este formula unghiului dintre o dreaptă și un plan?",
    "raspuns_asteptat": "sin phi = |A*l + B*m + C*n| / (sqrt(A^2 + B^2 + C^2) * sqrt(l^2 + m^2 + n^2))."
  },
  {
    "intrebare": "Care este definiția elipsei?",
    "raspuns_asteptat": "Se numeste elipsa locul geometric al punctelor din plan pentru care suma distantelor de la ele pana la doua puncte fixe F1 si F2, numite focare este constanta, egala cu 2a, presupunandu-se ca distanta dintre cele doua focare este 2c, unde c este un numar real pozitiv sau nul, verificand inegalitatea c < a."
  },
  {
    "intrebare": "Care este ecuația canonică a elipsei?",
    "raspuns_asteptat": "x2/a2 + y2/b2 = 1: (5.1.6) ... se numeste ecuatia canonica a elipsei."
  },
  {
    "intrebare": "Cum se definește excentricitatea elipsei?",
    "raspuns_asteptat": "Se numeste excentricitate a elipsei numarul real e = c/a."
  },
  {
    "intrebare": "Care este ecuația tangentei la elipsă într-un punct?",
    "raspuns_asteptat": "ecuatia tangentei intr-un punct al unei elipse se poate scrie prin dedublare. xx0/a2 + yy0/b2 = 1."
  },
  {
    "intrebare": "Care este definiția hiperbolei?",
    "raspuns_asteptat": "Se numeste hiperbola figura geometrica formata din toate punctele din plan pentru care valoarea absoluta a diferentei distantelor pana la punctele fixe F1 si F2 este constanta, egala cu 2a."
  },
  {
    "intrebare": "Care este ecuația canonică a hiperbolei?",
    "raspuns_asteptat": "x2/a2 - y2/b2 = 1: (5.2.6) ... se numeste ecuatia canonica a hiperbolei."
  },
  {
    "intrebare": "Care este ecuația tangentei la hiperbolă într-un punct?",
    "raspuns_asteptat": "se obtine xx0/a2 - yy0/b2 = 1 pentru ecuatia tangentei in punctul M0(x0, y0) al hiperbolei."
  },
  {
    "intrebare": "Care este definiția parabolei?",
    "raspuns_asteptat": "Se numeste parabola locul geometric al punctelor din plan care sunt egal departate de o dreapta fixa, numita directoare si de un punct fix F, numit focar."
  },
  {
    "intrebare": "Care este ecuația canonică a parabolei?",
    "raspuns_asteptat": "y2 = 2px: (5.3.3) Ecuatia (5.3.3) se numeste ecuatia canonica a parabolei de parametru p."
  },
  {
    "intrebare": "Care este ecuația tangentei la parabolă într-un punct?",
    "raspuns_asteptat": "ecuatia tangentei este: p(x - x0) + y0(y - y0) = 0 sau yy0 = p(x + x0)."
  },
  {
    "intrebare": "Care este ecuația unui elipsoid?",
    "raspuns_asteptat": "Un elipsoid este o suprafata de ecuatie x2/a2 + y2/b2 + z2/c2 = 1."
  },
  {
    "intrebare": "Care este ecuația conului de gradul al doilea?",
    "raspuns_asteptat": "Se numeste con de gradul al doilea multimea punctelor din spatiu ale caror coordonate relative la un sistem ortonormat verifica o ecuatie de forma x2/a2 + y2/b2 - z2/c2 = 0."
  },
  {
    "intrebare": "Care este ecuația hiperboloidului cu o pânză?",
    "raspuns_asteptat": "Se numeste hiperboloid cu o panza locul geometric al punctelor din spatiu ale caror coordonate relativ la un sistem rectangular verifica ecuatia x2/a2 + y2/b2 - z2/c2 = 1."
  }

  # ... (restul datelor tale pot fi adăugate aici sau încărcate) ...
  # Pentru test, las doar 2 exemple, dar tu pune tot lista ta in variabila asta!
]

# Am adăugat aici lista ta completă ca să poți da copy-paste direct la tot scriptul dacă vrei,
# dar ideal e să citești dintr-un json extern dacă lista e mare.
# (Momentan scriptul va rula pe lista raw_data de mai sus. Asigură-te că pui tot JSON-ul tău în ea)

# Prompt-ul care transformă "Dicționarul" în "Profesor"
SYSTEM_PROMPT = """Ești un Profesor de Matematică carismatic și prietenos.
Scopul tău este să rescrii o definiție tehnică într-o explicație clară, vorbită, potrivită pentru un tutorial video.
REGULI:
1. Păstrează corectitudinea matematică.
2. Folosește un ton conversațional ("Uite...", "Practic...", "Cu alte cuvinte...").
3. Evită formulele matematice complexe scrise cu simboluri ASCII (ex: nu scrie 'lambda*mu', scrie 'produsul dintre lambda și mu').
4. Răspunsul trebuie să fie scurt (maxim 2-3 fraze).
5. Nu folosi liste cu bullet points (*), scrie totul ca un paragraf fluid.
"""

def clean_text_for_tts(text):
    # O curățare simplă pentru a ajuta TTS-ul
    text = text.replace("*", "") 
    text = text.replace("_", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print(f"[INFO] Se încarcă modelul Llama pentru procesare...")
    try:
        llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=-1, 
            n_ctx=4096,
            verbose=False
        )
    except Exception as e:
        print(f"[EROARE] Nu s-a putut încărca modelul: {e}")
        return

    processed_data = []

    print(f"[INFO] Începem procesarea a {len(raw_data)} perechi...")

    for i, item in enumerate(raw_data):
        q = item['intrebare']
        a_tech = item['raspuns_asteptat']

        # Construim prompt-ul pentru Llama
        user_content = f"ÎNTREBARE: {q}\nDEFINIȚIE TEHNICĂ: {a_tech}\n\nRescrie răspunsul pentru video:"
        
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7, # Puțină creativitate pentru stil
        )
        
        a_style = output['choices'][0]['message']['content'].strip()
        a_style = clean_text_for_tts(a_style)

        print(f"\n[{i+1}/{len(raw_data)}] {q}")
        print(f"   -> Original: {a_tech[:50]}...")
        print(f"   -> Profesor: {a_style}")

        # Formatul pentru Llama 3 Fine-Tuning (Formatul Chat)
        # Acesta este formatul standard acceptat de unsloth/huggingface
        entry = {
            "messages": [
                {"role": "system", "content": "Ești un asistent universitar prietenos expert în geometrie."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a_style}
            ]
        }
        processed_data.append(entry)

    # Salvare în format JSONL (JSON Lines)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"\n[SUCCESS] Dataset generat: {OUTPUT_FILE}")
    print("Acum poți folosi acest fișier pentru Fine-Tuning!")

if __name__ == "__main__":
    # Înlocuiește raw_data cu lista ta completă de sus înainte să rulezi!
    # Pentru test, am lăsat doar primele 2 elemente în lista de mai sus.
    # Copiază tot array-ul tău în variabila `raw_data` din cod.
    
    # Sfat: Ca să meargă repede, pune raw_data = [ ... paste la tot json-ul tău ... ]
    
    # Aici suprascriu raw_data cu datele tale (pentru a fi sigur că rulezi pe toate)
    # Dă paste la JSON-ul tău complet în variabila de mai jos dacă vrei să rulezi acum:
    # raw_data = ... 
    
    main()