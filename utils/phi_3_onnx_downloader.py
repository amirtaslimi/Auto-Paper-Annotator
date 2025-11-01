from huggingface_hub import snapshot_download
# model link = https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32
# onnx example = https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3-qa.py
# --- Configuration ---
repo_id = "microsoft/Phi-3-mini-4k-instruct-onnx"
folder_to_download = "cuda/cuda-int4-rtn-block-32"
local_download_path = f"./{repo_id.split('/')[-1]}-{folder_to_download}" # e.g., ./stable-diffusion-xl-base-1.0-unet

# --- Download the folder ---
print(f"Downloading folder '{folder_to_download}' from repo '{repo_id}'...")

snapshot_download(
    repo_id=repo_id,
    allow_patterns=f"{folder_to_download}/*",  # The glob pattern to match files in the folder
    local_dir=local_download_path,
    local_dir_use_symlinks=False, # Set to False to download files directly instead of symlinking
    # Set to 'dataset', 'space' or 'model' if not a model
    # repo_type="dataset"
)

print(f"Folder downloaded successfully to: {local_download_path}")