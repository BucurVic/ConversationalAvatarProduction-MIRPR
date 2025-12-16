import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset

# --- CONFIGURARE ---
INPUT_DATASET = "dataset_finetuning_ready.jsonl"
OUTPUT_MODEL_NAME = "Llama-3-Profesor-Geometrie" # Numele final al fișierului
MAX_SEQ_LENGTH = 2048
DTYPE = None # None = auto detect (float16 pt majoritatea GPU-urilor, bfloat16 pt Ampere/Hopper)
LOAD_IN_4BIT = True # OBLIGATORIU True pentru a încăpea în VRAM de consumator

def main():
    # 1. Verificăm dacă avem GPU
    if not torch.cuda.is_available():
        print("[EROARE] Nu am detectat GPU NVIDIA! Antrenarea necesită CUDA.")
        return

    print(f"[INFO] Se încarcă modelul de bază Llama 3...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # 2. Adăugăm adaptoarele LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",    
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None, 
    )

    # 3. Încărcare Dataset Local
    if not os.path.exists(INPUT_DATASET):
        print(f"[EROARE] Nu găsesc fișierul {INPUT_DATASET}. Asigură-te că e în același folder.")
        return

    print(f"[INFO] Se încarcă datele din {INPUT_DATASET}...")
    dataset = load_dataset("json", data_files=INPUT_DATASET, split="train")

    # Formatare prompt
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 4. Antrenare
    print("\n[INFO] Începe antrenarea (poate dura câteva minute)...")
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = 2, # Scade la 1 dacă primești eroare de VRAM
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Pentru 50 de exemple, 60 pași e ok.
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    trainer_stats = trainer.train()
    print("[SUCCESS] Antrenare finalizată.")

    # 5. Export GGUF
    print(f"\n[INFO] Se convertește modelul în format GGUF (Q4_K_M)...")
    # Această funcție va descărca automat llama.cpp și va face conversia
    try:
        model.save_pretrained_gguf(OUTPUT_MODEL_NAME, tokenizer, quantization_method = "q4_k_m")
        print(f"\n[DONE] Model salvat cu succes în folderul: {OUTPUT_MODEL_NAME}")
        print(f"Caută fișierul .gguf în acel folder și mută-l în proiectul tău RAG.")
    except Exception as e:
        print(f"[EROARE la export GGUF] {e}")

if __name__ == "__main__":
    main()