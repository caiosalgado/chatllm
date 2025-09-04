from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Any
import re
import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM


class ChatLLM:
    """
    ChatLLM
    -------
    - Carrega tokenizer/modelo (Transformers) uma vez (reuso).
    - Mantém um histórico interno de mensagens (system/user/assistant).
    - Exponde helpers de histórico simples e expressivos:
        history(), clear_history(), pop_last_turn(),
        set_system(), get_system(), append_user(), append_assistant().
    - Gera respostas via ask() e atualiza o histórico automaticamente.
    """

    def __init__(
        self,
        model: str,
        system: str = "",
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        generation_defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model
        self._system: Optional[str] = system or None
        # somente user/assistant ficam aqui; system é armazenado separadamente
        self._messages: List[Dict[str, str]] = []

        # Carrega Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

        # Defaults de geração (podem ser sobrescritos a cada ask(...))
        self.generation_defaults: Dict[str, Any] = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 256,
        }
        if generation_defaults:
            self.generation_defaults.update(generation_defaults)

        # Descobre se o chat template parece suportar "thinking"
        tmpl = getattr(self.tokenizer, "chat_template", "") or ""
        self._template_supports_thinking = ("<think" in tmpl) or ("enable_thinking" in tmpl)

    # ------------------------------------------------------------------
    # HISTÓRICO — API principal (os 7 métodos que você pediu)
    # ------------------------------------------------------------------

    def history(self) -> List[Dict[str, str]]:
        """
        Retorna a lista de mensagens no formato: [{"role": "...", "content": "..."}]
        Inclui 'system' (se existir) seguido por user/assistant em ordem cronológica.
        """
        msgs: List[Dict[str, str]] = []
        if self._system:
            msgs.append({"role": "system", "content": self._system})
        msgs.extend(self._messages)
        return msgs

    def clear_history(self, keep_system: bool = True) -> None:
        """
        Zera o histórico. Se keep_system=False, também remove a mensagem de system.
        """
        self._messages.clear()
        if not keep_system:
            self._system = None

    def pop_last_turn(self) -> None:
        """
        Remove o último 'turno' (par user + assistant). Se só houver uma mensagem
        do assistant no final (por exemplo, inserida manualmente), ela também é removida.
        """
        if not self._messages:
            return
        # remove possível resposta final do assistente
        if self._messages and self._messages[-1]["role"] == "assistant":
            self._messages.pop()
        # remove a pergunta correspondente do usuário (se existir)
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages.pop()

    def set_system(self, content: str) -> None:
        """Define/atualiza a mensagem de system (use string vazia para remover)."""
        self._system = content or None

    def get_system(self) -> Optional[str]:
        """Retorna a mensagem de system atual (ou None)."""
        return self._system

    def append_user(self, content: str) -> None:
        """Adiciona manualmente uma mensagem de usuário ao histórico."""
        self._messages.append({"role": "user", "content": content})

    def append_assistant(self, content: str) -> None:
        """Adiciona manualmente uma mensagem do assistente ao histórico."""
        self._messages.append({"role": "assistant", "content": content})

    # ------------------------------------------------------------------
    # GERAÇÃO
    # ------------------------------------------------------------------

    def ask(
        self,
        text: str,
        *,
        enable_thinking: Optional[bool] = None,
        add_generation_prompt: bool = True,
        **gen_kwargs: Any,
    ) -> Tuple[str, str]:
        """
        Adiciona o texto do usuário ao histórico, gera a resposta e salva a resposta.
        Retorna (thinking, content).

        Parâmetros comuns em gen_kwargs:
            do_sample=True/False, temperature, top_p, top_k,
            repetition_penalty, max_new_tokens, etc.
        """
        # 1) adiciona a pergunta do usuário ao histórico interno
        self.append_user(text)

        # 2) monta a lista completa de mensagens para o chat template
        msgs = self.history()

        # 3) decide se tenta 'thinking' (se o template não suportar, será ignorado)
        use_thinking = False
        if enable_thinking is True:
            use_thinking = True
        elif enable_thinking is None:
            use_thinking = self._template_supports_thinking

        # 4) aplica chat template → string (não tokeniza aqui)
        prompt_text = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **({"enable_thinking": True} if use_thinking else {}),
        )

        # 5) tokeniza e envia para o device do modelo
        model_inputs = self.tokenizer(
            [prompt_text],
            return_tensors="pt",
            add_special_tokens=False,  # já formatamos via chat_template
        ).to(self.model.device)

        # 6) configura geração (defaults + overrides)
        cfg = dict(self.generation_defaults)
        cfg.update(gen_kwargs)

        # 7) gera
        out_ids = self.model.generate(**model_inputs, **cfg)
        # tokens novos (apenas a saída)
        new_tokens = out_ids[0][len(model_inputs.input_ids[0]):]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 8) separa thinking/content
        thinking, content = self._split_thinking_and_content(decoded)

        # 9) persiste no histórico a resposta final (sem thinking)
        self.append_assistant(content)
        return thinking, content

    # Utilitário: separa <think>...</think> do conteúdo
    @staticmethod
    def _split_thinking_and_content(text: str) -> Tuple[str, str]:
        tag_pairs = [
            (r"<think>", r"</think>"),
            (r"<\|think\|>", r"<\|/think\|>"),
        ]
        for start_tag, end_tag in tag_pairs:
            start = re.search(start_tag, text, flags=re.IGNORECASE)
            end = re.search(end_tag, text, flags=re.IGNORECASE)
            if start and end and end.start() >= start.end():
                thinking = text[start.end(): end.start()].strip()
                content = (text[:start.start()] + text[end.end():]).strip()
                return thinking, content
        return "", text.strip()

    # ------------------------------------------------------------------
    # OPCIONAIS PRÁTICOS (persistência/inspeção)
    # ------------------------------------------------------------------

    def export_json(self, path: str | Path) -> None:
        """Salva o histórico (incluindo system) em JSON."""
        obj = {"system": self._system, "messages": self._messages}
        Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def import_json(self, path: str | Path) -> None:
        """Carrega e substitui o histórico a partir de um JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._system = data.get("system") or None
        msgs = data.get("messages") or []
        # validação simples
        self._messages = [
            {"role": m["role"], "content": m["content"]}
            for m in msgs
            if isinstance(m, dict) and "role" in m and "content" in m
        ]


class HistoryBook:
    """
    Opcional: gerenciador simples de históricos nomeados.
    Útil para guardar/alternar conversas fora da instância do ChatLLM.
    """

    def __init__(self) -> None:
        # id -> {"system": Optional[str], "messages": List[dict]}
        self._store: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, *, system: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None) -> None:
        self._store[key] = {
            "system": system or None,
            "messages": list(messages) if messages else [],
        }

    def get(self, key: str) -> List[Dict[str, str]]:
        slot = self._store.get(key, {"system": None, "messages": []})
        out: List[Dict[str, str]] = []
        if slot["system"]:
            out.append({"role": "system", "content": slot["system"]})
        out.extend(slot["messages"])
        return out

    def append(self, key: str, *, role: str, content: str) -> None:
        slot = self._store.setdefault(key, {"system": None, "messages": []})
        if role == "system":
            slot["system"] = content
        else:
            slot["messages"].append({"role": role, "content": content})

    def clear(self, key: str, *, keep_system: bool = True) -> None:
        slot = self._store.setdefault(key, {"system": None, "messages": []})
        slot["messages"].clear()
        if not keep_system:
            slot["system"] = None
