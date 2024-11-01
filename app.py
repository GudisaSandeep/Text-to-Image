import os
from openai import AzureOpenAI
import json
import gradio as gr
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO

load_dotenv()
client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://speak-bot.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

def generate_image(prompt):
    result = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1
    )
    image_url = json.loads(result.model_dump_json())['data'][0]['url']
    
    # Download and convert image to PIL format
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image

# Create Gradio Interface
demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your image prompt"),
    outputs=gr.Image(label="Generated Image", type="pil"),
    title="DALL-E 3 Image Generator",
    description="Generate images using Azure OpenAI's DALL-E 3"
)

# Production deployment settings
demo.queue()
app = demo.app

if __name__ == "__main__":
    demo.launch()
