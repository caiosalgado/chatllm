from chatllm import ChatLLM

def test_history_api_basics():
    chat = ChatLLM("dummy-model", system="sys0")

    # system getters/setters
    assert chat.get_system() == "sys0"
    chat.set_system("sys1")
    assert chat.get_system() == "sys1"

    # append e leitura
    chat.append_user("u1")
    chat.append_assistant("a1")
    hist = chat.history()
    assert hist[0]["role"] == "system" and hist[0]["content"] == "sys1"
    assert hist[-2:] == [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]

    # pop last turn
    chat.pop_last_turn()
    hist2 = chat.history()
    assert hist2[-1]["role"] != "assistant"
    assert all(m["role"] != "assistant" for m in hist2[-2:])

    # clear (mantém system por padrão)
    chat.clear_history()
    assert chat.history() == [{"role": "system", "content": "sys1"}]

    # clear removendo system
    chat.clear_history(keep_system=False)
    assert chat.history() == []
