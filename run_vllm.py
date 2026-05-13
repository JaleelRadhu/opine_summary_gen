import subprocess
import sys
import os

# Configuration
MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
# We alias the model to "gpt-3.5-turbo-0125" so the other scripts (which default to this name) 
# can use it without needing explicit --model_name arguments everywhere.
PORT = 8000
def main():
    print(f"Initializing vLLM server...")
    print(f"Model: {MODEL_REPO}")
    print(f"Address: http://0.0.0.0:{PORT}/v1")
    
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0",
        "--model", MODEL_REPO,
        "--port", str(PORT),
        "--trust-remote-code", # Often required for Qwen models
        "--gpu-memory-utilization", "0.6"
        # "--gpu-memory-utilization", "0.6"
    ]
    
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        subprocess.run(command, check=True, env=env)
    except KeyboardInterrupt:
        print("\nvLLM server stopped.")

if __name__ == "__main__":
    main()