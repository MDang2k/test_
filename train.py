import tensorflow as tf
import cProfile
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint 
from src.model import get_model
from src.dataset import read_data



initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

def run(model):

    data = read_data('./data')

    train_loader = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
    validation_loader = tf.data.Dataset.from_tensor_slices((data[2], data[3]))

    batch_size = 1

    train_dataset = (
        train_loader.shuffle(len(data[0]))
            .batch(batch_size)
            .prefetch(2)
        )

    validation_dataset = (
            validation_loader.shuffle(len(data[2]))
            .batch(batch_size)
            .prefetch(2)
        )

    data = train_dataset.take(1)
    images, labels = list(data)[0]
    images = images.numpy()
    image = images[0]
    
    
    #Complie model
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"]
    )
    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    
    #Train model
    model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=5,
            batch_size=20,
            callbacks=[checkpoint_cb, early_stopping_cb],
        )

if __name__ == '__main__':
    
    model = get_model()
    run(model)
    
    