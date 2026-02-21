import cv2
import numpy as np
import tensorflow as tf

# =========================
# SETTINGS
# =========================

IMG_SIZE = 320
CLASS_NAMES = ["mouse"]   # 🔥 Change if you trained more classes
CONF_THRESHOLD = 0.5

# =========================
# LOAD TFLITE MODEL
# =========================

interpreter = tf.lite.Interpreter(model_path="ssd_custom.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully!")
print("Output shapes:")
for out in output_details:
    print(out["shape"])

# =========================
# START CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # =========================
    # PREPROCESS
    # =========================
    image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # =========================
    # GET OUTPUTS SAFELY
    # =========================
    output0 = interpreter.get_tensor(output_details[0]['index'])
    output1 = interpreter.get_tensor(output_details[1]['index'])

    # Determine which is bbox (shape = 4)
    if output0.shape[-1] == 4:
        bbox = output0[0]
        class_probs = output1[0]
    else:
        bbox = output1[0]
        class_probs = output0[0]

    # =========================
    # CLASS + CONFIDENCE
    # =========================
    class_id = np.argmax(class_probs)
    confidence = class_probs[class_id]

    if confidence > CONF_THRESHOLD:

        xmin, ymin, xmax, ymax = bbox

        # Convert from normalized (0-1) to actual pixels
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)

        # Clamp values inside frame
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        # Draw box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
        cv2.putText(frame, label,
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    # =========================
    # SHOW FRAME
    # =========================
    cv2.imshow("SSD Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()