import os
import subprocess
import sys

def print_big_message(message):
    """Affiche un message dans une grande boîte."""
    print(f"\n{'-' * len(message)}")
    print(message)
    print(f"{'-' * len(message)}\n")

def run_cmd(cmd, assert_success=False, environment=False):
    """Exécute une commande shell et affiche la sortie."""
    print(f"Exécution de la commande: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, text=True)
    if assert_success and result.returncode != 0:
        raise Exception(f"Command failed: {cmd}")
    return result

def get_user_choice(prompt, options):
    """Affiche une liste d'options et récupère la sélection de l'utilisateur."""
    print(f"{prompt}")
    for key, value in options.items():
        print(f"{key}: {value}")
    choice = input("Sélectionnez une option : ").upper()
    return choice

def install_webui():
    """Installe WebUI en fonction de la configuration du GPU."""
    # Demander à l'utilisateur quel GPU il utilise
    choice = get_user_choice(
        "Quel est votre GPU ?",
        {
            'A': 'NVIDIA',
            'B': 'AMD (Linux/MacOS only. Requires ROCm SDK 5.6 on Linux)',
            'C': 'Apple M Series',
            'D': 'Intel Arc (IPEX)',
            'E': 'NVIDIA CUDA 3.5 (Use system-installed PyTorch)',
            'N': 'None (I want to run models in CPU mode)'
        },
    )

    gpu_choice_to_name = {
        "A": "NVIDIA",
        "B": "AMD",
        "C": "APPLE",
        "D": "INTEL",
        "E": "NVIDIA_CUDA_35",
        "N": "NONE"
    }

    selected_gpu = gpu_choice_to_name[choice]

    if selected_gpu == "NVIDIA_CUDA_35":
        print_big_message("Using system-installed PyTorch for NVIDIA CUDA 3.5.")
        # Ne pas installer PyTorch, utiliser uniquement les dépendances nécessaires
        requirements_file = "requirements_cuda35.txt"
        update_requirements(initial_installation=True, requirements_file=requirements_file, pull=False)
        return

    # Si l'option NVIDIA CUDA 3.5 n'est pas choisie, on continue avec l'installation standard
    print_big_message("Installing PyTorch and other dependencies.")
    update_requirements(initial_installation=True)

def update_requirements(initial_installation=False, pull=True, requirements_file=None):
    """Met à jour les dépendances selon le fichier requirements.txt spécifié."""
    if not requirements_file:
        # Fichier requirements par défaut
        torver = torch_version()
        is_cuda = '+cu' in torver
        is_rocm = '+rocm' in torver
        is_cpu = '+cpu' in torver

        if is_rocm:
            base_requirements = "requirements_amd.txt"
        elif is_cpu:
            base_requirements = "requirements_cpu_only.txt"
        else:
            base_requirements = "requirements.txt"

        requirements_file = base_requirements

    print_big_message(f"Installing webui requirements from file: {requirements_file}")
    
    # Charger les dépendances à partir du fichier
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()

    # Installer les dépendances
    run_cmd(f"python -m pip install -r {requirements_file} --upgrade", assert_success=True, environment=True)

def torch_version():
    """Retourne la version actuelle de PyTorch installée."""
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "torch not installed"

def main():
    print_big_message("Installation de WebUI")
    install_webui()

if __name__ == "__main__":
    main()
