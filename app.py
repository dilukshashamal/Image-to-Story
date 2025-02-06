import os
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline

# Load environment variables 
load_dotenv(find_dotenv())

# Initialize pipelines once to improve efficiency
image_to_text_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

hf_token = os.getenv("HUGGINGFACE_API_TOKEN")

text_generator_pipeline = pipeline(
    "text-generation",
    model="microsoft/phi-2",  #
    trust_remote_code=True,
    token=hf_token  
)

# Function to extract text from an image
def img2text(url):
    text_output = image_to_text_pipeline(url)[0]["generated_text"]  # Fixed key name
    print("Extracted Text:", text_output)
    return text_output

# Function to generate a story from a given text
def generate_story(text):

    story_output = text_generator_pipeline(
        text, 
        max_length=200,  
        num_return_sequences=1,  
        do_sample=True, 
        top_k=50,  
        top_p=0.9  
    )

    story = story_output[0]["generated_text"]
    print("Generated Story:", story)
    return story


def img_to_story(image_url):
    extracted_text = img2text(image_url)  # Step 1: Extract text
    story = generate_story(extracted_text)  # Step 2: Generate story from extracted text
    return story

# Example Usage
image_path = "photo.png" 
final_story = img_to_story(image_path)
print("\nFinal Story:\n", final_story)




