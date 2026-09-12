import ollama

response = ollama.chat(
    model="qwen3.5:2b",
    messages=[
        {
            "role": "user",
            "content": "What is an LSTM? Explain in 2 sentences."
        }
    ]
)

print("FULL RESPONSE:")
print(response)

print("\nMESSAGE:")
print(response["message"])

print("\nCONTENT:")
print(response["message"]["content"])