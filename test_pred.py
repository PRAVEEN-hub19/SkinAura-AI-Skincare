from inference import predict_skin_type
from PIL import Image
import numpy as np

# create a dummy image
img = Image.fromarray(np.zeros((96, 96, 3), dtype=np.uint8))
try:
    label, confidence = predict_skin_type(img)
    print(f"Success: {label}, {confidence}")
except Exception as e:
    print(f"Error: {e}")
