import openai
import requests
import os

# Set up your OpenAI API key
openai.api_key = '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'

# Function to generate an image from a text prompt
def generate_image(prompt, size="1024x1024"):
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size=size,
        )
        # Get the URL of the generated image
        image_url = response['data'][0]['url']
        
        # Download the image
        image_response = requests.get(image_url)
        image_name = "generated_image.png"
        
        # Save the image
        with open(image_name, "wb") as image_file:
            image_file.write(image_response.content)
        
        print(f"Image generated and saved as {image_name}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Input prompt from the user
    prompt = input("Enter the prompt to generate an image: ")
    
    # Call the function to generate the image
    generate_image(prompt)
