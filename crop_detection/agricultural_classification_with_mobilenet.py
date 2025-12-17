# Auto-generated from agricultural-classification-with-mobilenet.ipynb
# Notes:
# - Working directory is set to this script's folder so relative paths work.
# - Kaggle download and /kaggle paths are disabled for local run.
# - EPOCHS can be overridden via environment variable EPOCHS (default=1 for quick smoke test).

import os
from pathlib import Path

# Ensure relative paths (like 'Agricultural-crops') resolve correctly
os.chdir(Path(__file__).parent)

# Fast/long training control
EPOCHS = int(os.getenv("EPOCHS", "1"))

# ==== Cell: Imports ====
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from zipfile import ZipFile

# ==== Cell: Kaggle download (disabled for local) ====
# !kaggle datasets download mdwaquarazam/agricultural-crops-image-classification

# ==== Cell: Import Dataset (zip extract) ====
# If a local zip exists, extract; otherwise skip
if Path('agricultural-crops-image-classification.zip').exists():
    with ZipFile('agricultural-crops-image-classification.zip') as zip:
        zip.extractall()

# ==== Cell: Data Preprocessing ====
DATA_DIR = 'Agricultural-crops'

train_datagen = ImageDataGenerator(
    rescale=1/255,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rotation_range=5,
    horizontal_flip=True,
    validation_split=0.1
)

test_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.1)

# ==== Cell: Directory flows ====
train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    subset='training',
    batch_size=32,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    subset='validation',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ==== Cell: Take sample batches ====
train_batch1 = next(train_gen)
train_images_batch1 = train_batch1[0]
train_labels_batch1 = train_batch1[1]

test_batch1 = next(test_gen)
test_images_batch1 = test_batch1[0]
test_labels_batch1 = test_batch1[1]

# ==== Cell: Build label dicts ====
train_indices = train_gen.class_indices
train_labels_dict = {value: key for key, value in train_indices.items()}
test_indices = test_gen.class_indices
test_labels_dict = {value: key for key, value in test_indices.items()}

# ==== Cell: Visualize dataset (optional) ====
# To avoid blocking scripts in headless environments, you can disable plotting.
if os.getenv('SHOW_PLOTS', '0') == '1':
    np.random.seed(88888)
    idx = np.random.choice(train_images_batch1.shape[0], size=25, replace=False)
    selected_imgs = [train_images_batch1[i] for i in idx]
    selected_labels = [np.argmax(train_labels_batch1[i]) for i in idx]
    plt.figure(figsize=(18, 18))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(selected_imgs[i])
        plt.title(train_labels_dict[selected_labels[i]], fontsize=18)
        plt.axis('off')
    plt.tight_layout(pad=0.8)
    plt.show()

# ==== Cell: Load MobileNet base ====
try:
    mobilenet = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
except Exception as e:
    print(f"Warning: failed to load ImageNet weights: {e}. Falling back to random init.")
    mobilenet = MobileNet(input_shape=(224, 224, 3), include_top=False, weights=None)
mobilenet.summary()

# ==== Cell: Modeling ====
tf.keras.backend.clear_session()
for layer in mobilenet.layers:
    layer.trainable = False

mobilenet_output = mobilenet.output
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(mobilenet_output)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(30, activation='softmax')(x)

model = tf.keras.Model(inputs=mobilenet.input, outputs=output)
model.summary()

# ==== Cell: Compile ====
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.9)
checkpoint_path = 'best_crop_detection_model.keras'
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

class stopEarly(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get('accuracy', 0) >= 0.9 and logs.get('val_accuracy', 0) >= 0.8:
            self.model.stop_training = True

stop_early = stopEarly()

# ==== Cell: Train (epochs controlled by EPOCHS env var) ====
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    callbacks=[reduceLR, stop_early, checkpoint_cb]
)

# ==== Print final train/val accuracy from history ====
final_train_acc = None
final_val_acc = None
if history.history:
    # Keras typically uses 'accuracy'/'val_accuracy'; fall back to 'acc' if needed
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc' if 'acc' in history.history else None
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc' if 'val_acc' in history.history else None
    if acc_key:
        final_train_acc = history.history[acc_key][-1]
    if val_acc_key:
        final_val_acc = history.history[val_acc_key][-1]

if final_train_acc is not None:
    print(f"Final Training Accuracy (last epoch): {final_train_acc:.4f}")
if final_val_acc is not None:
    print(f"Final Validation Accuracy (last epoch): {final_val_acc:.4f}")

# ==== Cell: Evaluate ====
test_loss, test_accuracy = model.evaluate(test_gen)
print(f'Test Accuracy: {test_accuracy:.4f}')

# ==== Save final model and class indices for future use ====
final_model_path = 'crop_detection_model.keras'
model.save(final_model_path)
print(f'Saved final model to: {final_model_path}')
print(f'Saved best checkpoint to: {checkpoint_path}')

# Save class indices mapping for inference
import pickle
with open('class_indices.pkl', 'wb') as f:
    pickle.dump(train_gen.class_indices, f)
print('Saved class indices to: class_indices.pkl')

# ==== Cell: Plot metrics (optional) ====
if os.getenv('SHOW_PLOTS', '0') == '1':
    plt.figure(figsize=(18, 8))
    plt.subplot(121)
    plt.title('Accuracy')
    plt.plot(history.history.get('accuracy', []), label='Training')
    plt.plot(history.history.get('val_accuracy', []), label='Testing')
    plt.legend()

    plt.subplot(122)
    plt.title('Loss')
    plt.plot(history.history.get('loss', []), label='Training')
    plt.plot(history.history.get('val_loss', []), label='Testing')
    plt.legend()
    plt.show()

# ==== Cell: Visualize predictions (optional) ====
predictions = model.predict(test_images_batch1)
predicted_labels = np.argmax(predictions, axis=-1)
true_labels = np.argmax(test_labels_batch1, axis=-1)

if os.getenv('SHOW_PLOTS', '0') == '1':
    np.random.seed(88888)
    idx = np.random.choice(len(test_images_batch1), size=25, replace=False)
    selected_test_imgs = [test_images_batch1[i] for i in idx]
    selected_predicted_labels = [predicted_labels[i] for i in idx]
    selected_true_labels = [true_labels[i] for i in idx]

    plt.figure(figsize=(18, 18))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(selected_test_imgs[i])
        plt.title(f"Pred: {test_labels_dict[selected_predicted_labels[i]]}\nTrue: {test_labels_dict[selected_true_labels[i]]}", fontsize=18)
        plt.axis('off')
    plt.tight_layout(pad=0.8)
    plt.show()

# ==== Cell: Single image prediction (disabled by default) ====
# Enable by setting SINGLE_IMAGE=1 and provide a valid LOCAL_IMAGE path
if os.getenv('SINGLE_IMAGE', '0') == '1':
    from PIL import Image

    def preprocess_image(image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    local_image = os.getenv('LOCAL_IMAGE', '')
    if local_image and Path(local_image).exists():
        img = plt.imread(local_image)
        processed_image = preprocess_image(img)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        print(f'Predicted Class: {predicted_class}')
    else:
        print('SINGLE_IMAGE=1 set but LOCAL_IMAGE not provided or not found; skipping single image prediction.')
