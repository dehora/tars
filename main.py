import sys
import anthropic


def chat(message: str) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": message}],
    )
    return response.content[0].text


def main():
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
    else:
        message = input("tars> ")

    print(chat(message))


if __name__ == "__main__":
    main()
