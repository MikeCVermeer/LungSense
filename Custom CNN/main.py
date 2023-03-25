import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import time

os.system('cls' if os.name == 'nt' else 'clear')

print("LungSense - Custom CNN model trainer\n")

# Check the command-line arguments
if len(sys.argv) < 2 or sys.argv[1].lower() not in ["fast", "slow", "custom"]:
    print("Usage: python main.py <mode> [<custom_cpu_percent>]")
    print("Where <mode> can be 'fast', 'slow', or 'custom'")
    print("If <mode> is 'custom', provide <custom_cpu_percent> (1-100)\n")

    print("Custom mode allows you to specify how much of the CPU to use for training the model")
    print("Fast mode uses all available CPU cores at 100%")
    print("Slow mode uses only 10% of the CPU cores")
    print("Slow mode is useful if you want to run other programs while training the model\n")

    print("Note: The model can possibly take much longer to train in slow mode")
    print("Note: The training will not necessarily be way faster with a higher cpu allocation as speed does not progress linearly along with cpu percentage allocated (depends on your CPU)\n")
    sys.exit(1)

mode = sys.argv[1].lower()

if mode == "custom":
    if len(sys.argv) != 3:
        print("Usage: python main.py custom <custom_cpu_percent> (1-100)")
        sys.exit(1)
    try:
        custom_cpu_percent = int(sys.argv[2])
        if custom_cpu_percent < 1 or custom_cpu_percent > 100:
            raise ValueError()
    except ValueError:
        print("Error: <custom_cpu_percent> should be an integer(number) between 1 and 100")
        sys.exit(1)

# Set the maximum percentage of the CPU that TensorFlow can use
if mode == "slow":
    max_cpu_percent = 10
elif mode == "custom":
    max_cpu_percent = custom_cpu_percent

if mode in ["slow", "custom"]:
    # Get the number of CPU cores
    num_cores = os.cpu_count()

    # Configure TensorFlow to use the specified percentage of the CPU
    intra_op_threads = int(num_cores * max_cpu_percent / 100)
    inter_op_threads = int(num_cores * max_cpu_percent / 100)

    tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)

# Try to estimate the training time before starting the actual training
def estimate_training_time(model, train_generator, validation_generator, epochs, batch_size):
    # Measure the time taken for the specified number of steps and epochs
    start_time = time.time()

    train_steps = 5
    val_steps = 5
    estimation_epochs = 2

    for epoch in range(estimation_epochs):
        for step in range(train_steps):
            x, y = next(train_generator)
            model.train_on_batch(x, y)
        for step in range(val_steps):
            x, y = next(validation_generator)
            model.test_on_batch(x, y)

    end_time = time.time()
    step_time = end_time - start_time

    # Calculate the number of steps per epoch
    nb_train_samples = len(train_generator.filenames)
    nb_validation_samples = len(validation_generator.filenames)
    train_steps_per_epoch = nb_train_samples // batch_size
    validation_steps_per_epoch = nb_validation_samples // batch_size

    # Estimate the total training time
    total_epochs = epochs
    total_steps = (train_steps_per_epoch + validation_steps_per_epoch) * total_epochs
    total_time = step_time * total_steps / ((train_steps + val_steps) * estimation_epochs)  # Divide by the product of steps and epochs used in estimation

    return total_time / 60

# Set the dataset directory paths
train_data_dir = 'chest_xray/train'
validation_data_dir = 'chest_xray/val'

# Set the image dimensions and batch size
img_width, img_height = 200, 200
batch_size = 32

# Create data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Set the number of training and validation samples
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)

# Train the model
epochs = 50

model_checkpoint = ModelCheckpoint('pneumonia_cnn_augmented.h5', save_best_only=True, monitor='val_loss')

# Estimate the training time before starting the actual training
print("\nEstimating training time...")
print("Note: This can take a while depending on your computer specs\n")
estimated_time = estimate_training_time(model, train_generator, validation_generator, epochs, batch_size)
os.system('cls' if os.name == 'nt' else 'clear')
print(f"Estimated training time: {estimated_time:.2f} minutes")
print("Note: Training time can deviate alot from the estimate based on cpu load during the training\n")
print("Training starting now...\n\n")

history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[model_checkpoint]
)

# Save the final model
model.save('pneumonia_cnn_augmented_final.h5')