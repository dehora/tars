import argparse

import anthropic
import ollama


CLAUDE_MODELS = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
    "opus": "claude-opus-4-6",
}

DEFAULT_MODEL = "ollama:gemma3:12b"


def parse_model(model_str: str) -> tuple[str, str]:
    provider, _, model = model_str.partition(":")
    if not model:
        raise ValueError(f"Invalid model format '{model_str}', expected provider:model")
    return provider, model


def chat_anthropic(messages: list[dict], model: str) -> str:
    resolved = CLAUDE_MODELS.get(model, model)
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=resolved,
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text


def chat_ollama(messages: list[dict], model: str) -> str:
    response = ollama.chat(model=model, messages=messages)
    return response.message.content


def chat(messages: list[dict], provider: str, model: str) -> str:
    if provider == "claude":
        return chat_anthropic(messages, model)
    if provider == "ollama":
        return chat_ollama(messages, model)
    raise ValueError(f"Unknown provider: {provider}")


def repl(provider: str, model: str):
    messages = []
    print(f"tars [{provider}:{model}] (ctrl-d to quit)")
    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input.strip():
            continue
        messages.append({"role": "user", "content": user_input})
        reply = chat(messages, provider, model)
        messages.append({"role": "assistant", "content": reply})
        print(f"tars> {reply}")


def main():
    parser = argparse.ArgumentParser(prog="tars", description="tars AI assistant")
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help="provider:model (e.g. ollama:gemma3:12b, claude:sonnet)",
    )
    parser.add_argument("message", nargs="*", help="message for single-shot mode")
    args = parser.parse_args()

    provider, model = parse_model(args.model)

    if args.message:
        message = " ".join(args.message)
        print(chat([{"role": "user", "content": message}], provider, model))
    else:
        repl(provider, model)


if __name__ == "__main__":
    main()
