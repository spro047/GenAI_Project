from huggingface_hub import hf_hub_download
import os

repo_id = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
filename = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
local_dir = r"d:\GenAI_Project\graphRAG\models"

print(f"Downloading {filename} from {repo_id} to {local_dir}...")
hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("Download complete.")
