# ChatLLM

Simple chat LLM wrapper with history helpers for Hugging Face transformers.

## Installation

### Via Git (recommended for development)

```bash
uv pip install "git+https://github.com/caiosalgado/chatllm@main"
```

### Via PyPI (when published)

```bash
uv pip install chatllm
```

### Local development

```bash
git clone https://github.com/caiosalgado/chatllm.git
cd chatllm
uv venv
uv sync
```

## Quick Start

```python
from chatllm import ChatLLM

# Initialize with any Hugging Face model
chat = ChatLLM("google/gemma-3-4b-it", system="You are a helpful assistant.")

# Ask a question
thinking, answer = chat.ask("What is the capital of Brazil?", temperature=0.7)
print(f"Answer: {answer}")
if thinking:
    print(f"Thinking: {thinking}")

# Access conversation history
history = chat.history()
print(f"Total messages: {len(history)}")

# Manage history
chat.clear_history()  # Clear all messages
chat.pop_last_turn()  # Remove last user+assistant exchange

# Update system message
chat.set_system("You are a coding expert.")
```

## Features

- **Simple API**: Easy-to-use chat interface with automatic history management
- **History Helpers**: Built-in methods to manage conversation history
- **Thinking Support**: Automatic detection and extraction of thinking blocks
- **Flexible**: Works with any Hugging Face model
- **Type Hints**: Full type annotation support
- **Export/Import**: Save and load conversation history as JSON

## API Reference

### ChatLLM Class

#### Constructor
```python
ChatLLM(
    model: str,                    # Hugging Face model name
    system: str = "",             # System prompt
    device_map: str = "auto",      # Device mapping strategy
    torch_dtype: str = "auto",     # PyTorch dtype
    trust_remote_code: bool = True,
    generation_defaults: dict = None
)
```

#### Methods

- `ask(text, **kwargs) -> Tuple[str, str]`: Generate response and update history
- `history() -> List[Dict[str, str]]`: Get conversation history
- `clear_history(keep_system=True)`: Clear conversation history
- `pop_last_turn()`: Remove last user+assistant exchange
- `set_system(content)`: Update system message
- `get_system() -> Optional[str]`: Get current system message
- `append_user(content)`: Manually add user message
- `append_assistant(content)`: Manually add assistant message
- `export_json(path)`: Save history to JSON file
- `import_json(path)`: Load history from JSON file

### HistoryBook Class (Optional)

Manage multiple named conversations:

```python
from chatllm import HistoryBook

book = HistoryBook()
book.set("chat1", system="Be helpful", messages=[...])
conversation = book.get("chat1")
```

## Development

### Setup

```bash
uv venv
uv sync
```

### Testing

```bash
pytest
```

### Building

```bash
uv build
```

### Publishing

```bash
UV_PUBLISH_TOKEN=your_token uv publish
```

## License

MIT License - see LICENSE file for details.
