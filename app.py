import os
from openai import AzureOpenAI
import json
import gradio as gr

client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://speak-bot.openai.azure.com/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

def generate_image(prompt):
    result = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1
    )
    image_url = json.loads(result.model_dump_json())['data'][0]['url']
    return image_url

# Create Gradio Interface
demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your image prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="DALL-E 3 Image Generator",
    description="Generate images using Azure OpenAI's DALL-E 3"
)

if __name__ == "__main__":
    demo.launch()
