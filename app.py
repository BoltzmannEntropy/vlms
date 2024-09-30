import os
import base64
import io
import sqlite3
import torch
import gradio as gr
import pandas as pd
from PIL import Image
import requests
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import traceback
from tqdm import tqdm 
import zipfile

# Define constants for vikhyatk/moondream2 model
MOON_DREAM_MODEL_ID = "vikhyatk/moondream2"
MOON_DREAM_REVISION = "2024-08-26"

# Define constants for the Qwen2-VL models
QWEN2_VL_MODELS = [
    'Qwen/Qwen2-VL-7B-Instruct',
    'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4',
    'OpenGVLab/InternVL2-1B',
    'Qwen/Qwen2-VL-72B',
]

# List of models to use (combining unique entries from available models and QWEN2_VL_MODELS)
available_models = [
    *QWEN2_VL_MODELS,  # Expands the QWEN2_VL_MODELS list into the available_models
    'microsoft/Phi-3-vision-128k-instruct',
    'vikhyatk/moondream2'
]

# List of available Hugging Face datasets
dataset_options = [
    "gokaygokay/panorama_hdr_dataset",  
    "OpenGVLab/CRPE"  
]

# List of text prompts to use
text_prompts = [
    "Provide a detailed description of the image contents, including all visible objects, people, activities, and extract any text present within the image using Optical Character Recognition (OCR). Organize the extracted text in a structured table format with columns for original text, its translation into English, and the language it is written in.",
    "Offer a thorough description of all elements within the image, from objects to individuals and their activities. Ensure any legible text seen in the image is extracted using Optical Character Recognition (OCR). Provide an accurate narrative that encapsulates the full content of the image.",    
    "Create a four-sentence caption for the image. Start by specifying the style and type, such as painting, photograph, or digital art. In the next sentences, detail the contents and the composition clearly and concisely. Use language suited for prompting a text-to-image model, separating descriptive terms with commas instead of 'or'. Keep the description direct, avoiding interpretive phrases or abstract expressions",
]

# SQLite setup
# def init_db():
#     conn = sqlite3.connect('image_outputs.db')
#     cursor = conn.cursor()
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS image_outputs (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             image BLOB,
#             prompt TEXT,
#             output TEXT,
#             model_name TEXT
#         )
#     ''')
#     conn.commit()
#     conn.close()

def image_to_binary(image_path):
    with open(image_path, 'rb') as file:
        return file.read()

# def store_in_db(image_path, prompt, output, model_name):
#     conn = sqlite3.connect('image_outputs.db')
#     cursor = conn.cursor()
#     image_blob = image_to_binary(image_path)
#     cursor.execute('''
#         INSERT INTO image_outputs (image, prompt, output, model_name)
#         VALUES (?, ?, ?, ?)
#     ''', (image_blob, prompt, output, model_name))
#     conn.commit()
#     conn.close()

# Function to encode an image to base64 for HTML display
def encode_image(image):
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_str}" style="max-width:500px;"/>'

# Function to load and display images from the panorama_hdr_dataset
def load_dataset_images(dataset_name, num_images):
    try:
        dataset = load_dataset(dataset_name, split='train')
        images = []
        for i, item in enumerate(dataset[:num_images]):
            if 'image' in item:
                img = item['image']
                print (type(img))
                encoded_img = encode_image(img)
                metadata = f"Width: {img.width}, Height: {img.height}"
                if 'hdr' in item:
                    metadata += f", HDR: {item['hdr']}"
                images.append(f"<div style='display: inline-block; margin: 10px; text-align: center;'><h3>Image {i+1}</h3>{encoded_img}<p>{metadata}</p></div>")
        if not images:
            return "No images could be loaded from this dataset. Please check the dataset structure."
        return "".join(images)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()

# Function to generate output
def generate_output(model, processor, prompt, image, model_name, device):
    try:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        if model_name in QWEN2_VL_MODELS:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_bytes},
                        {"type": "text", "text": prompt},
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[Image.open(io.BytesIO(image_bytes))],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
            response_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response_text

        elif model_name == 'microsoft/Phi-3-vision-128k-instruct':
            messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(prompt, [image], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=1024)
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
            return response_text

        elif model_name == 'vikhyatk/moondream2':
            tokenizer = AutoTokenizer.from_pretrained(MOON_DREAM_MODEL_ID, revision=MOON_DREAM_REVISION)
            enc_image = model.encode_image(image)
            response_text = model.answer_question(enc_image, prompt, tokenizer)
            return response_text
    except Exception as e:
        return f"Error during generation with model {model_name}: {e}"

# Function to list and encode images from a directory
def list_images(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            encoded_img = encode_image(image_path)
            images.append({
                "filename": filename,
                "image": encoded_img
            })
    return images

# Function to extract images from a ZIP file
# Function to extract images from a ZIP file
def extract_images_from_zip(zip_file):
    images = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                with zip_ref.open(file_info) as file:
                    try:
                        img = Image.open(file)
                        img = img.convert("RGB")  # Ensure the image is in RGB mode
                        encoded_img = img
                        images.append({
                            "filename": file_info.filename,
                            "image": encoded_img
                        })
                    except Exception as e:
                        print(f"Error opening image {file_info.filename}: {e}")
    return images

# Gradio interface function for running inference
def run_inference(model_names, dataset_input, num_images_input, prompts, device_map, torch_dtype, trust_remote_code,use_flash_attn, use_zip, zip_file):
    data = []

    torch_dtype_value = torch.float16 if torch_dtype == "torch.float16" else torch.float32
    device_map_value = "cuda" if torch.cuda.is_available() else "cpu" if device_map == "auto" else device_map

    model_processors = {}
    for model_name in model_names:
        try:
            if model_name in QWEN2_VL_MODELS:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name, 
                    torch_dtype=torch_dtype_value, 
                    device_map=device_map_value
                ).eval()
                processor = AutoProcessor.from_pretrained(model_name)
            elif model_name == 'microsoft/Phi-3-vision-128k-instruct':
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map=device_map_value, 
                    torch_dtype=torch_dtype_value, 
                    trust_remote_code=trust_remote_code, 
                    use_flash_attn=use_flash_attn
                ).eval()
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            elif model_name == 'vikhyatk/moondream2':
                model = AutoModelForCausalLM.from_pretrained(
                    MOON_DREAM_MODEL_ID, 
                    trust_remote_code=True, 
                    revision=MOON_DREAM_REVISION
                ).eval()
                processor = None  # No processor needed for this model

            model_processors[model_name] = (model, processor)

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")

    try:
        # Load images from the ZIP file if use_zip is True
        if use_zip:
            images = extract_images_from_zip(zip_file)
            print ("Number of images in zip:" , len(images))
            for img in tqdm(images):
                try:
                    img_data = img['image']
                    if not isinstance(img_data, str):
                        # Convert the Image object to a base64-encoded string
                        img_buffer = io.BytesIO()
                        img['image'].save(img_buffer, format="PNG")
                        img_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                            
                    img_data=f'<img src="data:image/png;base64,{img_data}" style="max-width:500px;"/>'

                    row_data = {"Image": img_data}  # Assuming encode_image is defined elsewhere
                    for model_name in model_names:
                        if model_name in model_processors:
                            model, processor = model_processors[model_name]
                            for prompt in prompts:
                                try:
                                    # Ensure image is defined
                                    image = img['image']
                                    response_text = generate_output(model, processor, prompt, image, model_name, device_map_value)
                                    row_data[f"{model_name}_Response_{prompt}"] = response_text
                                except Exception as e:
                                    row_data[f"{model_name}_Response_{prompt}"] = f"Error during generation with model {model_name}: {e}"
                                    traceback.print_exc()

                    data.append(row_data)
                except Exception as e:
                    print(f"Error processing image {img['filename']}: {e}")
                    traceback.print_exc()

        # Load the dataset if use_zip is False
        else:
            dataset = load_dataset(dataset_input, split='train')
            for i in tqdm(range(num_images_input)):
                if dataset_input == "OpenGVLab/CRPE":
                    image = dataset[i]['image']
                elif dataset_input == "gokaygokay/panorama_hdr_dataset":
                    image = dataset[i]['png_image']
                else:
                    image = dataset[i]['image']

                encoded_img = encode_image(image)
                row_data = {"Image": encoded_img}

                for model_name in model_names:
                    if model_name in model_processors:
                        model, processor = model_processors[model_name]
                        for prompt in prompts:
                            try:
                                response_text = generate_output(model, processor, prompt, image, model_name, device_map_value)
                                row_data[f"{model_name}_Response_{prompt}"] = response_text
                            except Exception as e:
                                row_data[f"{model_name}_Response_{prompt}"] = f"Error during generation with model {model_name}: {e}"

                data.append(row_data)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()

    return pd.DataFrame(data).to_html(escape=False)

def show_image(image):
    return image  # Simply display the selected image

# Gradio UI setup
def create_gradio_interface():
    css = """
      #output {
        height: 500px;
        overflow: auto;
      }
    """
    with gr.Blocks(css=css) as demo:
        # Title
        gr.Markdown("# VLM-Image-Analysis: A Vision-and-Language Modeling Framework.")
        gr.Markdown("""
                    - Handle a batch of images from a ZIP file OR
                    - Processes images from an HF DB 
                    - Compatible with png, jpg, jpeg, and webp formats                                        
                    - Compatibility with various AI models: Qwen2-VL-7B-Instruct, Qwen2-VL-2B-Instruct-GPTQ-Int4, InternVL2-1B, Qwen2-VL-72B, /Phi-3-vision-128k-instruct and moondream2""")

        # image_path = os.path.abspath("static/image.jpg")
        # gr.Image(value=image_path, label="HF Image", width=300, height=300)
        
        init_image = gr.Image(label="Selected Image", type="filepath")

        # Use gr.Examples to showcase a set of example images
        gr.Examples(
            examples=[
                ["static/image.jpg"],                
            ],
            inputs=[init_image],
            label="Example Images"
        )
        init_image.change(show_image, inputs=init_image, outputs=init_image)

        with gr.Tab("VLM model and Dataset selection"):
            gr.Markdown("### Dataset Selection: HF or from a ZIP file.")
            with gr.Accordion("Advanced Settings", open=True):
                with gr.Row():
                    # with gr.Column():  
                    use_zip_input = gr.Checkbox(label="Use ZIP File", value=False)
                    dataset_input = gr.Dropdown(choices=dataset_options, label="Select Dataset", value=dataset_options[1], visible=True)
                    num_images_input = gr.Radio(choices=[1, 5, 20], label="Number of Images", value=5)
                    zip_file_input = gr.File(label="Upload ZIP File of Images", file_types=[".zip"])
            gr.Markdown("### VLM Model Selection")
            with gr.Row():
                with gr.Column():  
                    models_input = gr.CheckboxGroup(choices=available_models, label="Select Models", value=available_models[4])
                    prompts_input = gr.CheckboxGroup(choices=text_prompts, label="Select Prompts", value=text_prompts[2])
                    submit_btn = gr.Button("Run Inference")
                
            with gr.Row():        
                output_display = gr.HTML(label="Results")

        with gr.Tab("GPU Device Settings"):
            device_map_input = gr.Radio(choices=["auto", "cpu", "cuda"], label="Device Map", value="auto")
            torch_dtype_input = gr.Radio(choices=["torch.float16", "torch.float32"], label="Torch Dtype", value="torch.float16")
            trust_remote_code_input = gr.Checkbox(label="Trust Remote Code", value=True)
            use_flash_attn = gr.Checkbox(label="Use flash-attn 2 (Ampere GPUs or newer.)", value=False)
            
                               
        def run_inference_wrapper(model_names, dataset_input, num_images_input, prompts, device_map, torch_dtype, trust_remote_code,use_flash_attn, use_zip, zip_file):
            return run_inference(model_names, dataset_input, num_images_input, prompts, device_map, torch_dtype, trust_remote_code,use_flash_attn, use_zip, zip_file)
        
        def toggle_dataset_visibility(use_zip):
            return gr.update(visible=not use_zip)
        
        submit_btn.click(
            fn=run_inference_wrapper, 
            inputs=[models_input, dataset_input, num_images_input, prompts_input, device_map_input, torch_dtype_input, trust_remote_code_input,use_flash_attn, use_zip_input, zip_file_input], 
            outputs=output_display
        )
        
        use_zip_input.change(
            fn=toggle_dataset_visibility, 
            inputs=use_zip_input, 
            outputs=dataset_input
        )

    demo.launch(debug=True, share=False)

if __name__ == "__main__":
    create_gradio_interface()