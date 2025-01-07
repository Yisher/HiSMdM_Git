import os
from PIL import Image
import torch
import numpy as np
from mod import Model

def load_model(model_path, device):
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image):
    image_np = np.array(image, dtype=np.float32)  
    image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0) 
    return image_tensor

def postprocess_tensor(tensor):
    tensor = tensor.squeeze(0).squeeze(0)  
    image_np = tensor.numpy().astype(np.uint16) 
    image = Image.fromarray(image_np)
    return image

def test(model, input_image, device):
    input_tensor = preprocess_image(input_image).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = postprocess_tensor(output_tensor.cpu())
    return output_image

def main():
    model_path = './model/UNet_195.pt'
    input_tiff_path = './experimental_data/input.tif'
    output_tiff_path = './Output/denoise_result.tif'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)

    #
    input_tiff = Image.open(input_tiff_path)

    # 
    image_stack = []

    frame_index = 0
    try:
        while True:
            input_image = input_tiff.copy()  # 
            output_image = test(model, input_image, device)  # 
            image_stack.append(output_image)  # 
            frame_index += 1
            input_tiff.seek(frame_index)  # 
    except EOFError:
        pass  # 

    # 
    if image_stack:
        image_stack[0].save(output_tiff_path, save_all=True, append_images=image_stack[1:], compression="tiff_deflate")
        print(f"Processed TIFF stack saved to {output_tiff_path}")
    else:
        print("No frames were processed and saved.")

if __name__ == '__main__':
    main()
