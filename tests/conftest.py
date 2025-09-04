import pytest

# Stubs leves que imitam só o necessário do HF
class FakeBatch:
    def __init__(self):
        # parecido com BatchEncoding: precisa ter input_ids e suportar **kwargs
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
    chat_template = "<think></think>"  # indica suporte a 'thinking'
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        assert isinstance(messages, list) and messages, "messages precisa ser lista"
        return "PROMPT"
    def __call__(self, texts, return_tensors="pt", add_special_tokens=False):
        assert isinstance(texts, list) and isinstance(texts[0], str)
        return FakeBatch()
    def decode(self, _tokens, skip_special_tokens=True):
        # devolve um texto com <think> para testar o split
        return "<think>diag</think>ok"

class FakeModel:
    def __init__(self):
        self.device = "cuda:0"
    def generate(self, input_ids, **kwargs):
        # retorna [concat(input_ids[0] + novos_tokens)]
        in_ids = input_ids[0]
        return [in_ids + [42, 43]]

@pytest.fixture(autouse=True)
def patch_hf(monkeypatch):
    # Monkeypatcha AutoTokenizer/AutoModelForCausalLM para não baixar nada
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: FakeModel())
    yield
