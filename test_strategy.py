import subprocess
import sys

# Percorso dell'interprete Python
python_executable = sys.executable
inputs = ["entropy", "loss", "random"]

for input_str in inputs:
    subprocess.run([python_executable, "test_agent.py", input_str])