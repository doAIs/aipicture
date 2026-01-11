from huggingface_hub import InferenceClient

# Step 1: 先去 https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
#        点击 "Agree and access repository"

client = InferenceClient(
    model="stabilityai/stable-diffusion-3-medium-diffusers",
    token="xxx"
)

try:
    image = client.text_to_image("Astronaut riding a horse", guidance_scale=7.5, num_inference_steps=28)
    image.save("astronaut_horse.png")
    print("✅ 图像已生成并保存为 astronaut_horse.png")
except Exception as e:
    print("❌ 错误:", str(e))