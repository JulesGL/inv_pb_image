from PIL import Image
from torchvision import transforms

def local_image_to_tensor(image_path, img_size=256):
    # Load the local image using PIL
    img = Image.open(image_path)

    # Define the transformation to convert the image to a PyTorch tensor
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Apply the transformation
    tensor_image = transform(img).unsqueeze(0)  # Add batch dimension
    return tensor_image