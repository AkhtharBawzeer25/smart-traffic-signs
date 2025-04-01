# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-mjwbynemuKXB6G5A3xfRdpKzYKyghvP
"""

pip install torchvision

from torchvision.datasets import GTSRB
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
])

train_dataset = GTSRB(root='data', split='train', transform=transform, download=True)
test_dataset = GTSRB(root='data', split='test', transform=transform, download=True)

import matplotlib.pyplot as plt
import numpy as np

# Function to display sample images
def show_images(dataset, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i in range(num_images):
        image, label = dataset[i]  # Extract image and label
        image = image.permute(1, 2, 0).numpy()  # Convert Tensor to NumPy for visualization

        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(f"Class {label}")

    plt.show()

# ✅ Display images from the dataset
show_images(train_dataset)

import matplotlib.pyplot as plt
import numpy as np

# Function to display multiple images from the dataset
def display_more_images(dataset, num_images=50):
    rows = num_images // 10  # Arrange images in rows of 10
    fig, axes = plt.subplots(rows, 10, figsize=(15, rows * 2))

    for i in range(num_images):
        image, label = dataset[i]  # Extract image and label
        image = image.permute(1, 2, 0).numpy()  # Convert Tensor to NumPy

        row = i // 10  # Get row index
        col = i % 10   # Get column index
        axes[row, col].imshow(image)
        axes[row, col].axis("off")
        axes[row, col].set_title(f"Class {label}")

    plt.show()

# ✅ Display 50 images from GTSRB
display_more_images(train_dataset, num_images=50)

import matplotlib.pyplot as plt

# Function to display one image from each class
def display_images_from_each_class(dataset, num_classes=43):
    fig, axes = plt.subplots(5, 9, figsize=(18, 10))  # Adjusted for 43 classes
    class_found = set()

    for image, label in dataset:
        if label not in class_found:
            row = len(class_found) // 9
            col = len(class_found) % 9
            image = image.permute(1, 2, 0).numpy()  # Convert Tensor to NumPy
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
            axes[row, col].set_title(f"Class {label}")
            class_found.add(label)

        if len(class_found) == num_classes:
            break

    plt.show()

# ✅ Display One Image from Each Class
display_images_from_each_class(train_dataset)

import cv2
import os
import numpy as np
from torchvision.datasets import GTSRB
from torchvision import transforms
from PIL import Image

# Load GTSRB dataset
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
dataset = GTSRB(root="data", split='train', download=True, transform=transform)

# Create your own dataset folder
output_dir = "my_gtsrb_custom_dataset"
os.makedirs(output_dir, exist_ok=True)

def add_custom_noise(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy()  # Convert to HWC
    noise = np.random.normal(0, 0.1, img.shape)  # Gaussian noise
    img_noisy = np.clip(img + noise, 0, 1) * 255
    return img_noisy.astype(np.uint8)

# Save custom images
for idx, (img, label) in enumerate(dataset):
    img_noisy = add_custom_noise(img)
    label_dir = os.path.join(output_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)

    cv2.imwrite(os.path.join(label_dir, f"{idx}.png"), cv2.cvtColor(img_noisy, cv2.COLOR_RGB2BGR))

    if idx % 1000 == 0:
        print(f"Saved {idx} images...")

import torchvision.transforms.functional as TF
import random

def augment_image(img_tensor):
    if random.random() > 0.5:
        img_tensor = TF.hflip(img_tensor)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        img_tensor = TF.rotate(img_tensor, angle)
    return img_tensor

from torchvision.datasets import ImageFolder
from torchvision import transforms

custom_dataset = ImageFolder(root="my_gtsrb_custom_dataset",
                             transform=transforms.ToTensor())

import os
import cv2
import matplotlib.pyplot as plt

def display_custom_gtsrb_images(base_dir, num_classes=5, images_per_class=5):
    class_folders = sorted(os.listdir(base_dir))[:num_classes]  # Limit how many classes to show
    fig, axes = plt.subplots(num_classes, images_per_class, figsize=(images_per_class*2, num_classes*2))

    for i, class_name in enumerate(class_folders):
        class_path = os.path.join(base_dir, class_name)
        image_files = sorted(os.listdir(class_path))[:images_per_class]

        for j in range(images_per_class):
            img_path = os.path.join(class_path, image_files[j])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            if num_classes == 1:
                axes[j].imshow(img)
                axes[j].axis('off')
                axes[j].set_title(f"Class {class_name}")
            else:
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_ylabel(f"Class {class_name}", rotation=0, labelpad=30)

    plt.tight_layout()
    plt.show()

# ✅ Call the function
display_custom_gtsrb_images("my_gtsrb_custom_dataset", num_classes=5, images_per_class=5)

import numpy as np
import cv2
import os
from tqdm import tqdm

def is_dark(image, threshold=40):
    """
    Determines if the image is too dark based on average brightness.
    Threshold ~40 is a good start for RGB images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    return brightness < threshold

def clean_dark_images(input_dir, output_dir, threshold=40):
    os.makedirs(output_dir, exist_ok=True)
    classes = os.listdir(input_dir)

    for class_name in classes:
        input_class_path = os.path.join(input_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        for filename in tqdm(os.listdir(input_class_path), desc=f"Processing {class_name}"):
            img_path = os.path.join(input_class_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if not is_dark(img, threshold=threshold):
                cv2.imwrite(os.path.join(output_class_path, filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print("✅ Dark images removed.")

# Example usage
clean_dark_images("my_gtsrb_custom_dataset", "my_gtsrb_cleaned_dataset", threshold=40)

display_custom_gtsrb_images("my_gtsrb_cleaned_dataset", num_classes=5, images_per_class=5)

import os
import cv2
import matplotlib.pyplot as plt

def display_gtsrb_images_extended(base_dir, num_classes=10, images_per_class=10):
    class_folders = sorted(os.listdir(base_dir))[:num_classes]
    fig, axes = plt.subplots(num_classes, images_per_class, figsize=(images_per_class * 2, num_classes * 2))

    for i, class_name in enumerate(class_folders):
        class_path = os.path.join(base_dir, class_name)
        image_files = sorted(os.listdir(class_path))[:images_per_class]

        for j in range(images_per_class):
            img_path = os.path.join(class_path, image_files[j])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(f"Class {class_name}", rotation=0, labelpad=30)

    plt.tight_layout()
    plt.show()

# 👇 Adjust how many classes and images you want to see
display_gtsrb_images_extended("my_gtsrb_cleaned_dataset", num_classes=10, images_per_class=10)

import os
import cv2
import numpy as np

# Base folder of cleaned GTSRB dataset
base_dir = "my_gtsrb_cleaned_dataset"

images = []
labels = []
class_names = sorted(os.listdir(base_dir))

for label, class_name in enumerate(class_names):
    class_path = os.path.join(base_dir, class_name)
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)

images = np.array(images, dtype=np.float32)
labels = np.array(labels)
print(f"Images shape: {images.shape}, Labels shape: {labels.shape}, Classes: {len(class_names)}")

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Normalize images for CNN (0-1 range)
images_cnn = images / 255.0

# One-hot encode labels
labels_cat = to_categorical(labels, num_classes=len(class_names))

# Split
x_train, x_test, y_train, y_test = train_test_split(images_cnn, labels_cat, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')  # One output per class
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=20,
    batch_size=64
)

import matplotlib.pyplot as plt

# Accuracy & Loss Plot
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()
plt.show()

# Final test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Get predicted class labels
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot it with better styling
plt.figure(figsize=(12, 10))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="YlGnBu",     # Use a more subtle palette
            linewidths=0.5,
            linecolor='gray',
            square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Confusion Matrix - GTSRB Classifier", fontsize=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

target_classes = [1, 2]

# Convert from one-hot if needed
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(x_test), axis=1)

# Filter for only the selected classes
mask = np.isin(y_true, target_classes)
y_true_filtered = y_true[mask]
y_pred_filtered = y_pred[mask]

# Also make sure predictions are within the same target classes
mask_pred = np.isin(y_pred_filtered, target_classes)
y_true_filtered = y_true_filtered[mask_pred]
y_pred_filtered = y_pred_filtered[mask_pred]

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=target_classes)

# Label map (adjust to your actual GTSRB class labels)
class_labels = ['30 km/h', '50 km/h']  # Customize as needed

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Binary Confusion Matrix: Class 1 vs Class 2')
plt.show()

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(images, labels_cat, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

# One-hot encode labels
num_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes)

# Split into train/validation
x_train, x_val, y_train, y_val = train_test_split(images, labels_cat, test_size=0.2, random_state=42)

from sklearn.metrics import classification_report

# Convert one-hot to class indices
y_true = np.argmax(y_val, axis=1)
y_pred = np.argmax(model.predict(x_val), axis=1)

# Print report
print(classification_report(y_true, y_pred, digits=4))

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
for i, idx in enumerate(misclassified_idx[:10]):
    plt.subplot(2, 5, i + 1)

    img = x_val[idx]

    #  Check if in [-1, 1] range — common with tanh
    if img.min() < 0:
        img = (img + 1.0) / 2.0

    #  If accidentally scaled to 0–255 range
    if img.max() > 1.0:
        img = img / 255.0

    img = np.clip(img, 0, 1)  # Safety

    plt.imshow(img)
    plt.title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

import os

def find_folder(target_folder, root='.'):
    for dirpath, dirnames, filenames in os.walk(root):
        if target_folder in dirnames:
            return os.path.join(dirpath, target_folder)
    return None

# Replace with your folder name
target = 'my_gtsrb_cleaned_dataset'
result = find_folder(target)

if result:
    print(f"✅ Found '{target}' at: {result}")
else:
    print(f"❌ Folder '{target}' not found.")

import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_gtsrb_images(folder, img_size=(32, 32), max_per_class=500):
    images = []
    class_dirs = sorted(os.listdir(folder))
    for cls in class_dirs:
        class_path = os.path.join(folder, cls)
        if not os.path.isdir(class_path):
            continue
        count = 0
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                count += 1
                if count >= max_per_class:
                    break
    return np.array(images)

# 📂 Load from cleaned dataset
image_data = load_gtsrb_images("my_gtsrb_cleaned_dataset", img_size=(32, 32))

# 🔍 Print shape
print("Loaded images shape:", image_data.shape)

# 🔄 Normalize to [-1, 1] for tanh activation
image_data = image_data.astype('float32') / 127.5 - 1.0

# 📤 Split into train and validation
x_auto_train, x_auto_val = train_test_split(image_data, test_size=0.2, random_state=42)

print("Train shape:", x_auto_train.shape)
print("Val shape:", x_auto_val.shape)

from sklearn.model_selection import train_test_split

# First split: 85% train_val, 15% test
x_train_val, x_test = train_test_split(images, test_size=0.15, random_state=42)

# Second split: from train_val (85% of total), get 70% train and 15% val
x_train, x_val = train_test_split(x_train_val, test_size=0.1765, random_state=42)
# 0.1765 ≈ 15 / 85, so val ends up ~15% of total

# Check shapes
print("Train shape:", x_train.shape)
print("Val shape:", x_val.shape)
print("Test shape:", x_test.shape)

def add_gaussian_noise(images, mean=0.0, std=0.1):
    noisy = images + np.random.normal(loc=mean, scale=std, size=images.shape)
    noisy = np.clip(noisy, 0., 1.)
    return noisy

# Add noise to each split
x_train_noisy = add_gaussian_noise(x_train)
x_val_noisy = add_gaussian_noise(x_val)
x_test_noisy = add_gaussian_noise(x_test)

import os
import cv2
import numpy as np
from tqdm import tqdm

def load_all_images_from_folder(folder, image_size=(32, 32)):
    images = []
    class_folders = sorted(os.listdir(folder))

    for class_folder in class_folders:
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
                images.append(img)

    return np.array(images)

# 👇 Load and assign to all_images
all_images = load_all_images_from_folder("my_gtsrb_cleaned_dataset")
print("✅ Loaded all images:", all_images.shape)

from sklearn.model_selection import train_test_split

# Split into train+val and test
x_temp, x_auto_test = train_test_split(all_images, test_size=0.2, random_state=42)

# Split x_temp into train and val (so: 64% train, 16% val, 20% test)
x_auto_train, x_auto_val = train_test_split(x_temp, test_size=0.2, random_state=42)

print("Train shape:", x_auto_train.shape)
print("Val shape:  ", x_auto_val.shape)
print("Test shape: ", x_auto_test.shape)

noise_factor = 0.2

x_auto_train_noisy = x_auto_train + noise_factor * np.random.normal(0.0, 1.0, x_auto_train.shape)
x_auto_val_noisy = x_auto_val + noise_factor * np.random.normal(0.0, 1.0, x_auto_val.shape)
x_auto_test_noisy = x_auto_test + noise_factor * np.random.normal(0.0, 1.0, x_auto_test.shape)

# Clip to [0, 1]
x_auto_train_noisy = np.clip(x_auto_train_noisy, 0., 1.)
x_auto_val_noisy = np.clip(x_auto_val_noisy, 0., 1.)
x_auto_test_noisy = np.clip(x_auto_test_noisy, 0., 1.)

import matplotlib.pyplot as plt

n = 10  # Number of images to show
plt.figure(figsize=(20, 4))

for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_auto_val[i])
    plt.title("Original")
    plt.axis("off")

    # Noisy image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_auto_val_noisy[i])
    plt.title("Noisy")
    plt.axis("off")

plt.tight_layout()
plt.show()

from tensorflow.keras import layers, models

# Define the Autoencoder
def build_autoencoder():
    input_img = layers.Input(shape=(32, 32, 3))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    return autoencoder

# Build and compile
autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

# Summary
autoencoder.summary()

# Train the autoencoder
history = autoencoder.fit(
    x_auto_train_noisy, x_auto_train,
    epochs=30,
    batch_size=64,
    shuffle=True,
    validation_data=(x_auto_val_noisy, x_auto_val)
)

import matplotlib.pyplot as plt

# Predict on validation noisy set
x_val_denoised = autoencoder.predict(x_auto_val_noisy)

# Display
n = 10
plt.figure(figsize=(18, 6))
for i in range(n):
    # Original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_auto_val[i])
    plt.title("Original")
    plt.axis("off")

    # Noisy
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_auto_val_noisy[i])
    plt.title("Noisy")
    plt.axis("off")

    # Denoised
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(x_val_denoised[i])
    plt.title("Denoised")
    plt.axis("off")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Autoencoder Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
from sklearn.metrics import mean_squared_error

# Flatten images for MSE calculation
x_true_flat = x_auto_val.reshape((len(x_auto_val), -1))
x_pred_flat = x_val_denoised.reshape((len(x_val_denoised), -1))

# Compute MSE for each image, then take the mean
mse_per_image = np.mean(np.square(x_true_flat - x_pred_flat), axis=1)
mean_mse = np.mean(mse_per_image)

print(f"📉 Mean Squared Error (MSE) on validation set: {mean_mse:.6f}")

!pip install scikit-image

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Calculate average PSNR and SSIM across all validation images
psnr_scores = []
ssim_scores = []

for i in range(len(x_auto_val)):
    original = x_auto_val[i]
    denoised = x_val_denoised[i]

    psnr_val = psnr(original, denoised, data_range=1.0)
    ssim_val = ssim(original, denoised, channel_axis=-1, data_range=1.0)

    psnr_scores.append(psnr_val)
    ssim_scores.append(ssim_val)

avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores)

print(f"🔸 Average PSNR: {avg_psnr:.2f} dB")
print(f"🔸 Average SSIM: {avg_ssim:.4f}")













