import streamlit as st
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

def main():

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = RealESRGAN(device, scale=4)
  model.load_weights('/content/weights/RealESRGAN_x4plus.pth', download=True)

  st.title("Enhance Super Resolution GAN App")

  # Upload an image file
  img_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

  if img_file_buffer is not None:
      # Display the uploaded image
      st.image(img_buffer, caption="Uploaded Image", use_column_width=True)

      image = Image.open(img_buffer)
      img_array = np.array(image)
      
      sr_image = model.predict(image)
      
      st.image(sr_image, caption="Enhanced Image", use_column_width=True)

if __name__ == "__main__":
    main()
