#!/usr/bin/env python3
"""
Exemplo mínimo usando mocks (não baixa modelos reais)
Execute com: uv run python examples/minimal_mock.py
"""

import os
import sys

# Adiciona o diretório raiz ao path para importar o chatllm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configura variáveis de ambiente para usar stubs
os.environ['HF_HUB_OFFLINE'] = '1'

# Importa após configurar o ambiente
from chatllm import ChatLLM

# Monkey patch para usar stubs (simula o que o pytest faz)
import transformers

class FakeBatch:
    def __init__(self):
        self.input_ids = [[1, 2, 3]]
        self.attention_mask = [[1, 1, 1]]
    def to(self, *_args, **_kwargs):
        return self
    def __getitem__(self, key):
        return getattr(self, key)
    def keys(self):
        return ['input_ids', 'attention_mask']
    def items(self):
        return [('input_ids', self.input_ids), ('attention_mask', self.attention_mask)]

class FakeTokenizer:
    chat_template = "<think></think>"
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "PROMPT"
    def __call__(self, texts, return_tensors="pt", add_special_tokens=False):
        return FakeBatch()
    def decode(self, _tokens, skip_special_tokens=True):
        return "<think>This is a mock thinking block</think>This is a mock response"

class FakeModel:
    def __init__(self):
        self.device = "cpu"
    def generate(self, input_ids, **kwargs):
        in_ids = input_ids[0]
        return [in_ids + [42, 43]]

# Aplica os patches
transformers.AutoTokenizer.from_pretrained = lambda *a, **k: FakeTokenizer()
transformers.AutoModelForCausalLM.from_pretrained = lambda *a, **k: FakeModel()

# Agora podemos usar o ChatLLM
chat = ChatLLM("dummy-model", system="You are a helpful assistant.")

print("=== Exemplo Mínimo com Mocks ===")
print("Inicializando chat...")

# Faz uma pergunta
thinking, answer = chat.ask("Hello, how are you?", temperature=0.7, top_p=0.95, max_new_tokens=256)

print(f"\nPergunta: Hello, how are you?")
print(f"Resposta: {answer}")
if thinking:
    print(f"Thinking: {thinking}")

print(f"\nHistórico completo:")
for i, msg in enumerate(chat.history()):
    print(f"{i+1}. [{msg['role']}] {msg['content']}")
