import cv2
import numpy as np
import torch
from RealESRGAN import RealESRGAN
from PIL import Image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RealESRGAN(device, scale=4)
    model.load_weights('/content/weights/RealESRGAN_x4plus.pth', download=True)

    # Create a VideoCapture object to capture webcam feed (use 0 for default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame with the ESRGAN model
        pil_frame = Image.fromarray((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sr_image = model.predict(pil_frame)

        # Convert the enhanced image back to an OpenCV format for display
        enhanced_frame = np.array(sr_image)

        # Display the enhanced frame
        cv2.imshow("ESRGAN Enhanced Image", enhanced_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
