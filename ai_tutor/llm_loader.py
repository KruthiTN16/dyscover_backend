from ollama import chat

MODEL_NAME = "llama3:latest"

def ask_llm(prompt):
    """Send a prompt to the LLM and return the response text."""
    response = chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    # The ChatResponse object has a 'message' attribute containing content
    return response.message.content

def main():
    print("Type 'exit' to quit the chat.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat. Goodbye!")
            break
        try:
            output = ask_llm(user_input)
            print(f"{MODEL_NAME}: {output}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()

