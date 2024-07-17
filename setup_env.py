#!/usr/bin/env python3
# setup_venv.py

import venv
import subprocess
import sys
from pathlib import Path

def create_venv():
    venv_path = Path("venv")
    venv.create(venv_path, with_pip=True)
    return venv_path

def install_dependencies(venv_path):
    pip_path = venv_path / "bin" / "pip"
    subprocess.check_call([pip_path, "install", "-U", "pip"])
    subprocess.check_call([pip_path, "install",
                           "pydantic",
                           "aiolimiter",
                           "tenacity",
                           "tqdm",
                           "aiohttp",
                           "jsonlines",
                           "pyyaml"])

def generate_requirements(venv_path):
    pip_path = venv_path / "bin" / "pip"
    with open("requirements.txt", "w") as f:
        subprocess.check_call([pip_path, "freeze"], stdout=f)

def main():
    print("Creating virtual environment...")
    venv_path = create_venv()

    print("Installing dependencies...")
    install_dependencies(venv_path)

    print("Generating requirements.txt...")
    generate_requirements(venv_path)

    print("Setup complete. To activate the virtual environment, run:")
    print(f"source {venv_path}/bin/activate")

if __name__ == "__main__":
    main()
