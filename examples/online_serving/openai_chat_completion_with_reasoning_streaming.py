# SPDX-License-Identifier: Apache-2.0
"""
An example shows how to generate chat completions from reasoning models
like DeepSeekR1.

To run this example, you need to start the vLLM server with the reasoning 
parser:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
     --enable-reasoning --reasoning-parser deepseek_r1
```

Unlike openai_chat_completion_with_reasoning.py, this example demonstrates the
streaming chat completions feature.

The streaming chat completions feature allows you to receive chat completions
in real-time as they are generated by the model. This is useful for scenarios
where you want to display chat completions to the user as they are generated
by the model.

Here we do not use the OpenAI Python client library, because it does not support
`reasoning_content` fields in the response.
"""

import json

import requests

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

models = requests.get(
    f"{openai_api_base}/models",
    headers={
        "Authorization": f"Bearer {openai_api_key}"
    },
).json()
model = models["data"][0]["id"]

# Streaming chat completions
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]

response = requests.post(
    f"{openai_api_base}/chat/completions",
    headers={"Authorization": f"Bearer {openai_api_key}"},
    json={
        "model": model,
        "messages": messages,
        "stream": True
    },
)

print("client: Start streaming chat completions...")
printed_reasoning_content = False
printed_content = False
# Make the streaming request
if response.status_code == 200:
    # Process the streaming response
    for line in response.iter_lines():
        if line:  # Filter out keep-alive new lines
            # Decode the line and parse the JSON
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data:"):
                data = decoded_line[5:].strip()  # Remove "data:" prefix
                if data == "[DONE]":  # End of stream
                    print("\nclient: Stream completed.")
                    break
                try:
                    # Parse the JSON data
                    chunk = json.loads(data)
                    reasoning_content = chunk["choices"][0]["delta"].get(
                        "reasoning_content", "")
                    content = chunk["choices"][0]["delta"].get("content", "")

                    if reasoning_content:
                        if not printed_reasoning_content:
                            printed_reasoning_content = True
                            print("reasoning_content:", end="", flush=True)
                        print(reasoning_content, end="", flush=True)
                    elif content:
                        if not printed_content:
                            printed_content = True
                            print("\ncontent:", end="", flush=True)
                        # Extract and print the content
                        print(content, end="", flush=True)
                except json.JSONDecodeError:
                    print("Error decoding JSON:", decoded_line)
else:
    print(f"Error: {response.status_code} - {response.text}")
