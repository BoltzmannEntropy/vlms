from datasets import load_dataset
import matplotlib.pyplot as plt

# Specify dataset and number of images to load
dataset_name = "OpenGVLab/CRPE"  # Can switch to "OpenGVLab/CRPE"
num_images = 2  # Specify the number of images you want to load

# Load the dataset
dataset = load_dataset(dataset_name, split='train')

# Check if the dataset is 'OpenGVLab/CRPE' or 'gokaygokay/panorama_hdr_dataset' and handle accordingly
for i in range(num_images):
    if dataset_name == "gokaygokay/panorama_hdr_dataset":
        img = dataset[i]["png_image"]  # Access image for this dataset
    elif dataset_name == "OpenGVLab/CRPE":
        img = dataset[i]["image"]  # Access image for 'OpenGVLab/CRPE'
    
    # Display the image
    plt.imshow(img)
    plt.title(f"Image {i+1}")
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
