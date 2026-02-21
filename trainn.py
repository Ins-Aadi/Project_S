import os
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# =========================
# SETTINGS
# =========================

DATASET_PATH = "dataset"
IMAGES_PATH = "images"
ANNOTATIONS_PATH = "annotations"

IMG_SIZE = 320
BATCH_SIZE = 8
EPOCHS = 25

CLASS_NAMES = ["mouse"]   # 🔥 CHANGE if more classes
NUM_CLASSES = len(CLASS_NAMES)

# =========================
# LOAD DATA
# =========================

images = []
boxes = []
labels = []

for xml_file in os.listdir(ANNOTATIONS_PATH):

    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(ANNOTATIONS_PATH, xml_file))
    root = tree.getroot()

    filename = root.find("filename").text
    img_path = os.path.join(IMAGES_PATH, filename)

    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    h, w, _ = image.shape

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0

    for obj in root.findall("object"):

        class_name = obj.find("name").text
        if class_name not in CLASS_NAMES:
            continue

        xmin = float(obj.find("bndbox/xmin").text)
        ymin = float(obj.find("bndbox/ymin").text)
        xmax = float(obj.find("bndbox/xmax").text)
        ymax = float(obj.find("bndbox/ymax").text)

        # Normalize
        xmin /= w
        xmax /= w
        ymin /= h
        ymax /= h

        images.append(image)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_NAMES.index(class_name))

images = np.array(images, dtype=np.float32)
boxes = np.array(boxes, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

print("Total samples:", len(images))

# Split
X_train, X_val, y_train_boxes, y_val_boxes, y_train_labels, y_val_labels = train_test_split(
    images, boxes, labels, test_size=0.2
)

# =========================
# BUILD SSD MODEL
# =========================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)

bbox_output = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox")(x)
class_output = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="class")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=[bbox_output, class_output])

# =========================
# COMPILE
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        "bbox": "mse",
        "class": "sparse_categorical_crossentropy"
    },
    metrics={"class": "accuracy"}
)

# =========================
# TRAIN
# =========================

model.fit(
    X_train,
    {"bbox": y_train_boxes, "class": y_train_labels},
    validation_data=(X_val, {"bbox": y_val_boxes, "class": y_val_labels}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# =========================
# SAVE MODEL
# =========================

model.save("ssd_custom.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("ssd_custom.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model Saved!")