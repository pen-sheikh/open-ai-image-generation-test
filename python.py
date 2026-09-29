from openai import OpenAI
import base64

client = OpenAI(api_key="")

result = client.images.generate(
    model="gpt-image-2",
    prompt="A cinematic photograph of a futuristic city in Tokyo at night, neon lights, rain, highly detailed"
)

image_bytes = base64.b64decode(result.data[0].b64_json)

with open("generated.png", "wb") as f:
    f.write(image_bytes)

print("Image saved as generated.png")
