import tensorflow as tf
import cProfile
import matplotlib.pyplot as plt
from tensorflow import keras
from src.model import get_model
from src.dataset import read_data


initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

def run(model):

    best_f1 = 0
    df = read_data('./data')

    model = get_model(width=128, height=128, depth=64)

    train_loader = tf.data.Dataset.from_tensor_slices((df.x_train, df.y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((df.x_val, df.y_val))

    batch_size = 2

    train_dataset = (
        train_loader.shuffle(len(df.x_train))
            .batch(batch_size)
            .prefetch(2)
        )

    validation_dataset = (
            validation_loader.shuffle(len(df.x_val))
            .batch(batch_size)
            .prefetch(2)
        )


    data = train_dataset.take(1)
    images, labels = list(data)[0]
    images = images.numpy()
    image = images[0]
    

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"]
    )

        
    model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=10,
            batch_size=20
        )

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(["acc", "loss"]):
        ax[i].plot(model.history.history[metric])
        ax[i].plot(model.history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])