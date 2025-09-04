from chatllm import HistoryBook

def test_history_book_set_get_append_clear():
    book = HistoryBook()

    # cria sessão
    book.set("A", system="S", messages=[{"role":"user","content":"u"}])
    hist = book.get("A")
    assert hist[0]["role"] == "system" and hist[0]["content"] == "S"
    assert hist[1]["role"] == "user" and hist[1]["content"] == "u"

    # append assistant
    book.append("A", role="assistant", content="a")
    hist = book.get("A")
    assert hist[-1] == {"role":"assistant","content":"a"}

    # clear mantendo system
    book.clear("A", keep_system=True)
    hist = book.get("A")
    assert hist == [{"role":"system","content":"S"}]

    # agora remove system
    book.clear("A", keep_system=False)
    assert book.get("A") == []
