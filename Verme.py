# Shim de compatibilidade: a simulação principal agora vive em main.py.
# Rode com: python main.py  (Verme.py apenas redireciona)
import sys
print("Verme.py -> redirecionando para main.py  (use 'python main.py' diretamente)")
print(f"  args: {sys.argv}")
import main  # noqa: F401