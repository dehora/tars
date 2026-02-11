import sys
import anthropic


def chat(messages: list[dict]) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text


def repl():
    messages = []
    print("tars (ctrl-d to quit)")
    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input.strip():
            continue
        messages.append({"role": "user", "content": user_input})
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"tars> {reply}")


def main():
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        print(chat([{"role": "user", "content": message}]))
    else:
        repl()


if __name__ == "__main__":
    main()
