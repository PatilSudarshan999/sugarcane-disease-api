# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# dataset_path = r"C:\Users\Sudarshan\OneDrive\Desktop\CaneAi\SugarCane_Leafs_Dataset"
# img_size = (224, 224)
# batch_size = 32
# epochs = 15  # Can increase later for better accuracy

# # Data augmentation
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=20,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# train_gen = datagen.flow_from_directory(
#     dataset_path,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training'
# )

# val_gen = datagen.flow_from_directory(
#     dataset_path,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation'
# )

# # Transfer learning base
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# predictions = Dense(train_gen.num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# # Freeze base layers
# for layer in base_model.layers:
#     layer.trainable = False

# model.compile(optimizer=Adam(learning_rate=0.0001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Train
# model.fit(train_gen, validation_data=val_gen, epochs=epochs)

# # Save model
# model.save("sugarcane_disease_model.h5")
# print("Model trained and saved successfully!")


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================
# SETTINGS
# =========================
DATASET_PATH = r"C:\Users\Sudarshan\OneDrive\Desktop\CaneAi\Sugarcane_leafs_Dataset"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS_INITIAL = 20
EPOCHS_FINE = 15

# =========================
# DATA GENERATOR
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes

# =========================
# MODEL
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(256, 256, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3
)

# =========================
# TRAIN PHASE 1
# =========================
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_INITIAL,
    callbacks=[early_stop, reduce_lr]
)

# =========================
# FINE TUNE
# =========================
for layer in base_model.layers[-25:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE,
    callbacks=[early_stop, reduce_lr]
)

model.save("sugarcane_6class_model.h5")

print("✅ Training Completed")