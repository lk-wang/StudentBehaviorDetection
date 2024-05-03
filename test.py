from VideoProcessorGPUCVCUDA import VideoProcessor
import cv2
from datetime import datetime

processor = VideoProcessor()

image = cv2.imread("./original.jpg")

postprocessedImage = processor.processing(image)

print(postprocessedImage.shape)

cv2.imwrite(f"./output_{datetime.now()}.jpg",postprocessedImage)


