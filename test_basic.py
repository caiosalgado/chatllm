#!/usr/bin/env python3
"""
Teste básico para verificar se o pacote chatllm está funcionando
"""

import sys
from pathlib import Path

# Adiciona o diretório src ao path para importar o pacote
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatllm import ChatLLM, HistoryBook

def test_import():
    """Testa se as classes podem ser importadas corretamente"""
    print("✓ Importação bem-sucedida")
    print(f"ChatLLM: {ChatLLM}")
    print(f"HistoryBook: {HistoryBook}")
    print(f"Versão: {__import__('chatllm').__version__}")

def test_history_book():
    """Testa a classe HistoryBook"""
    book = HistoryBook()
    
    # Testa set/get
    book.set("test", system="You are a test assistant", messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ])
    
    conversation = book.get("test")
    print(f"✓ HistoryBook funcionando: {len(conversation)} mensagens")
    
    # Testa append
    book.append("test", role="user", content="How are you?")
    conversation = book.get("test")
    print(f"✓ Append funcionando: {len(conversation)} mensagens")

if __name__ == "__main__":
    print("=== Testando pacote chatllm ===")
    
    try:
        test_import()
        test_history_book()
        print("\n✓ Todos os testes passaram!")
        print("\nO pacote está pronto para uso!")
        
    except Exception as e:
        print(f"\n✗ Erro: {e}")
        sys.exit(1)
