#!/usr/bin/env python3
"""
Exemplo mínimo de uso do ChatLLM
"""

from chatllm import ChatLLM

# Inicializa com um modelo (não será baixado neste exemplo devido aos stubs)
chat = ChatLLM("dummy-model", system="You are a helpful assistant.")

# Faz uma pergunta
thinking, answer = chat.ask("ping", temperature=0.7, top_p=0.95, max_new_tokens=256)

print(f"Resposta: {answer}")
if thinking:
    print(f"Thinking: {thinking}")

print("\nHistórico:")
for msg in chat.history():
    print(f"{msg['role']}: {msg['content']}")
