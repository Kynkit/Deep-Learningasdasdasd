from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("1. Downloading CIFAR-100 dataset from fast mirror...")
dataset = load_dataset("uoft-cs/cifar100")

# Target fine class indices for Food Containers:
# 9 = bottle, 10 = bowl, 16 = can, 28 = cup, 61 = plate
TARGET_CLASSES = [9, 10, 16, 28, 61]
CLASS_NAMES = ["bottle", "bowl", "can", "cup", "plate"]
label_map = {orig_id: new_id for new_id, orig_id in enumerate(TARGET_CLASSES)}


def extract_split(split_name):
  imgs, lbls = [], []
  for item in dataset[split_name]:
    lbl = item.get("fine_label", item.get("label"))
    if lbl in TARGET_CLASSES:
      img_data = item.get("img", item.get("image"))
      imgs.append(np.array(img_data))
      lbls.append(label_map[lbl])
  return np.array(imgs, dtype="float32") / 255.0, np.array(lbls, dtype="int64")


x_train, y_train = extract_split("train")
x_test, y_test = extract_split("test")

print(f"Dataset ready: {len(x_train)} train images, {len(x_test)} test images.")

print("2. Designing CNN model...")
model = models.Sequential([
    layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
    layers.RandomRotation(0.1),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(5, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("3. Training model (30 epochs)...")
history = model.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_test, y_test),
)

print("4. Evaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%\n")

# 5. Plot and save curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["loss"], label="Training Loss")
ax1.plot(history.history["val_loss"], label="Validation Loss", linestyle="--")
ax1.set_title("Training & Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(history.history["accuracy"], label="Training Accuracy")
ax2.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    linestyle="--",
)
ax2.set_title("Training & Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("loss_accuracy_curves.png")
print("Graph saved as 'loss_accuracy_curves.png' inside your folder!")
plt.show()