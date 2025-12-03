from llama_cpp import Llama
import time

llm = Llama(
    model_path="./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=512,
    verbose=False
)

prompt = "Explain what a neural network is in one paragraph."

start = time.time()
output = llm(prompt, max_tokens=150)
end = time.time()

print(output["choices"][0]["text"])
print("Time:", end - start)
