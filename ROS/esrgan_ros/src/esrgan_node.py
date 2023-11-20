#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from RealESRGAN import RealESRGAN
from PIL import Image as PILImage
import signal
import logging

class ESRGANManager:
    def __init__(self, model_path='/path/to/weights/RealESRGAN_x4plus.pth'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RealESRGAN(device, scale=4)
        self.model.load_weights(model_path, download=True)

    def predict(self, pil_frame):
        return self.model.predict(pil_frame)

def capture_frame(cap, esrgan_manager, bridge, image_queue, rate):
    try:
        while not rospy.is_shutdown():
            ret, frame = cap.read()

            if not ret:
                break

            pil_frame = PILImage.fromarray(frame)

            sr_image = esrgan_manager.predict(pil_frame)
            enhanced_frame = np.array(sr_image)

            # Put the enhanced frame into the queue
            image_queue.put(enhanced_frame)

            rate.sleep()

    except Exception as e:
        logging.error(f"Error in capture_frame: {e}")
    finally:
        cap.release()

def publish_frame(image_queue, image_pub, rate):
    try:
        while not rospy.is_shutdown():
            # Get the enhanced frame from the queue
            enhanced_frame = image_queue.get()

            ros_enhanced_frame = CvBridge().cv2_to_imgmsg(enhanced_frame, encoding="bgr8")
            image_pub.publish(ros_enhanced_frame)

            rate.sleep()

    except Exception as e:
        logging.error(f"Error in publish_frame: {e}")

def shutdown_handler(signum, frame):
    rospy.signal_shutdown("Received shutdown signal")

def main():
    rospy.init_node('esrgan_node', anonymous=True)
    rate = rospy.Rate(30)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        esrgan_manager = ESRGANManager()

        cap = cv2.VideoCapture(0)
        image_queue = Queue()  # Use multiprocessing.Queue for inter-process communication
        image_pub = rospy.Publisher('esrgan_frame', Image, queue_size=10)

        # Start the capture_frame and publish_frame processes
        capture_process = Process(target=capture_frame, args=(cap, esrgan_manager, CvBridge(), image_queue, rate))
        capture_process.start()

        publish_process = Process(target=publish_frame, args=(image_queue, image_pub, rate))
        publish_process.start()

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    finally:
        capture_process.join()
        publish_process.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
