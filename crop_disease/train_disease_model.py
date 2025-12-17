"""
Disease Detection Model Training Script
Trains a model with >95% validation accuracy using 18 epochs
Saves model compatible with TensorFlow 2.16+ for app.py
"""
import os
import pickle
import json

# Set matplotlib backend to non-interactive to avoid import issues
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'Plant_Disease_Dataset')
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'valid')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

# Model parameters
IMG_SIZE = 128  # Keep same as existing model
BATCH_SIZE = 32
EPOCHS = 18
LEARNING_RATE = 0.001

# Output files
MODEL_OUT = os.path.join(BASE_DIR, 'trained_model_18epochs.h5')  # Use .h5 for compatibility
CLASS_INDICES_OUT = os.path.join(BASE_DIR, 'class_indices.pkl')
HISTORY_OUT = os.path.join(BASE_DIR, 'training_history_18epochs.json')

print(f"Training Configuration:")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Data: {DATA_ROOT}")

# Data generators - NO RESCALING (model will work with raw 0-255 values like main.py)
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()  # No augmentation for validation

# Load data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"\nClasses found: {num_classes}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Save class indices
with open(CLASS_INDICES_OUT, 'wb') as f:
    pickle.dump(train_generator.class_indices, f)
print(f"Saved class indices to: {CLASS_INDICES_OUT}")

# Build model - similar architecture to existing trained_model_new.keras
model = models.Sequential([
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third conv block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Fourth conv block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Dense layers
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_OUT,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

# Train model
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Save training history
history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
with open(HISTORY_OUT, 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"\nSaved training history to: {HISTORY_OUT}")

# Final evaluation
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"\nValidation Accuracy: {val_acc*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

if val_acc >= 0.95:
    print("\n✅ SUCCESS! Achieved >95% validation accuracy")
else:
    print(f"\n⚠️  Target not reached. Got {val_acc*100:.2f}%, need >95%")

print(f"\nModel saved to: {MODEL_OUT}")
print("Ready to use in app.py!")
