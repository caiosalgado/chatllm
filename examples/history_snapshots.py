#!/usr/bin/env python3
"""
Exemplo de uso do ChatLLM com histórico e snapshots
"""

from chatllm import ChatLLM, HistoryBook
from pathlib import Path

# Inicializa o chat
chat = ChatLLM("dummy-model", system="You are a coding expert.")

print("=== Conversa 1 ===")
thinking1, answer1 = chat.ask("What is Python?", temperature=0.7)
print(f"Q: What is Python?")
print(f"A: {answer1}")

thinking2, answer2 = chat.ask("Give me an example", temperature=0.7)
print(f"Q: Give me an example")
print(f"A: {answer2}")

print(f"\nTotal messages: {len(chat.history())}")

# Salva snapshot
chat.export_json("conversation1.json")
print("Snapshot salvo em conversation1.json")

# Limpa e inicia nova conversa
chat.clear_history()
print("\n=== Conversa 2 (histórico limpo) ===")

thinking3, answer3 = chat.ask("Explain list comprehensions", temperature=0.7)
print(f"Q: Explain list comprehensions")
print(f"A: {answer3}")

# Usa HistoryBook para gerenciar múltiplas conversas
book = HistoryBook()
book.set("python_basics", messages=chat.history())

# Carrega conversa anterior
chat.import_json("conversation1.json")
book.set("full_tutorial", messages=chat.history())

print(f"\nConversas salvas no HistoryBook:")
for name in ["python_basics", "full_tutorial"]:
    msgs = book.get(name)
    print(f"  {name}: {len(msgs)} mensagens")

# Limpa arquivos de exemplo
Path("conversation1.json").unlink(missing_ok=True)
