from chatllm import ChatLLM

def test_ask_updates_history_and_parses_thinking():
    chat = ChatLLM("dummy-model", system="S")
    thinking, content = chat.ask("hello", temperature=0.7, top_p=0.95, max_new_tokens=10)

    # stubs retornam "<think>diag</think>ok"
    assert thinking == "diag"
    assert content == "ok"

    hist = chat.history()
    # deve conter system, user("hello") e assistant("ok")
    assert hist[0]["role"] == "system"
    assert hist[-2:] == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "ok"}]
