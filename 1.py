import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

messages = []


def main():
    while True:
        user_input = input("You: ")

        if user_input.lower().strip() == "stop":
            print("Goodbye!")
            break

        messages.append({
            "role": "user",
            "content": user_input,
        })

        response = client.responses.create(
            model="openai/gpt-oss-20b",
            input=messages,
        )

        assistant_message = response.output_text
        messages.extend(response.output)

        print(f"Assistant: {assistant_message}")


if __name__ == "__main__":
    main()

