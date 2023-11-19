#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from RealESRGAN import RealESRGAN
from PIL import Image as PILImage
from multiprocessing import Process, Queue

class ESRGANManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ESRGANManager, cls).__new__(cls, *args, **kwargs)
            # Load the ESRGAN model only once
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RealESRGAN(device, scale=4)
        self.model.load_weights('/path/to/weights/RealESRGAN_x4plus.pth', download=True)

    def predict(self, pil_frame):
        return self.model.predict(pil_frame)

def capture_frames(cap, esrgan_manager, bridge, image_queue, rate):
    while not rospy.is_shutdown():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert to PIL Image directly without using cv2
        pil_frame = PILImage.fromarray(frame)

        sr_image = esrgan_manager.predict(pil_frame)
        enhanced_frame = np.array(sr_image)

        ros_enhanced_frame = bridge.cv2_to_imgmsg(enhanced_frame, encoding="bgr8")
        image_queue.put(ros_enhanced_frame)

        rate.sleep()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def publish_frames(image_queue, image_pub):
    while not rospy.is_shutdown():
        if not image_queue.empty():
            ros_enhanced_frame = image_queue.get()
            image_pub.publish(ros_enhanced_frame)

def main():
    rospy.init_node('esrgan_node', anonymous=True)
    rate = rospy.Rate(30)

    esrgan_manager = ESRGANManager()

    # Adjust capture settings based on your needs
    cap = cv2.VideoCapture(0)
    bridge = CvBridge()
    image_pub = rospy.Publisher('esrgan_frame', Image, queue_size=10)
    image_queue = Queue()

    capture_process = Process(target=capture_frames, args=(cap, esrgan_manager, bridge, image_queue, rate))
    capture_process.start()

    publish_process = Process(target=publish_frames, args=(image_queue, image_pub))
    publish_process.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

    capture_process.join()
    publish_process.join()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
