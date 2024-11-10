from huggingface_hub import InferenceClient

client = InferenceClient(api_key="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")

messages = [
	{ "role": "user", "content": s }
]

stream = client.chat.completions.create(
    model="HuggingFaceH4/zephyr-7b-beta", 
	messages=messages, 
	temperature=0.5,
	max_tokens=1024,
	top_p=0.7,
	stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content)