import subprocess
from modules import shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger


class LlamaCppModel:
    def __init__(self):
        self.executable_path = "/home/ia/llama.cpp/llama-cli"  # Replace this path with the location of your llama.cpp bianire

    def generate(self, prompt, state, callback=None):
        cmd = [
            self.executable_path,
            "-m", shared.args.model_path,
            "-p", prompt,
            "--tokens", str(state['max_new_tokens']),
            "--temperature", str(state['temperature']),
            "--top-p", str(state['top_p']),
            "--top-k", str(state['top_k']),
        ]
        if state['seed'] != -1:
            cmd.extend(["--seed", str(state['seed'])])

        logger.info(f"Executing command: {' '.join(cmd)}")

        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = process.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing llama-cli: {e.stderr}")
            return ""

        output_text = ""
        for line in output.splitlines():
            output_text += line
            if callback:
                callback(line)
        return output_text

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
