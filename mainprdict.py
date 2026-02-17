import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = 224
CLASS_NAMES = ['flat', 'mouse','mud', 'stairs']
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
image = cv2.imread("image.jpeg")
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0
image = np.expand_dims(image.astype(np.float32), axis=0)
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
prediction = CLASS_NAMES[np.argmax(output)]

print("Prediction:", prediction)
