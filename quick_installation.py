import os
import subprocess

# Install packages from requirements.txt
subprocess.check_call(["pip", "install", "-r", "requirements.txt"])

# Create folders
os.makedirs("Archive", exist_ok=True)
os.makedirs("history", exist_ok=True)
os.makedirs("GGUFs", exist_ok=True)


import os

try:
    os.system("huggingface-cli download TheBloke/Mixtral-8x7B-v0.1-GGUF mixtral-8x7b-v0.1.Q4_K_M.gguf --local-dir GGUFs")
except Exception as e:
    print("An error occurred while downloading the file.\n But do not panic, this is easy to go around:")
    print(f"Error message: {str(e)}")
    print("Please download the file manually to the GGUFs folder. we are using TheBloke/Mixtral-8x7B-v0.1-GGUF mixtral-8x7b-v0.1.Q4_K_M.gguf from Huggingface for now as the default llm")
    print("Note: The download process could take a while depending on your internet connection and the file size.")