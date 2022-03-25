import streamlit as st
import tensorflow as tf
import numpy as np
import cProfile
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint 
from src.model import get_model
from src.dataset import read_data

tf.executing_eagerly()


siteHeader = st.container()
dataExploration = st.container()
modelTraining = st.container()
prediction = st.container()


data = read_data('./data')

train_loader = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
validation_loader = tf.data.Dataset.from_tensor_slices((data[2], data[3]))

x_val = data[2]

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

with siteHeader:
    st.title('Welcome to the Alzheimer\'s disease Detection project!')
    st.text('In this project I build a CNN model to classify normal and abnormal MRI image \nAnd I worked with the dataset from http://adni.loni.usc.edu/')

with dataExploration:
    st.header('Dataset: MRI images dataset')
    slider_input = st.slider('Choose which image to display?', min_value=1, max_value=len(train_dataset),value=1)
    height_input = st.slider('Select height:', min_value=1, max_value=64,value=1)
    data = train_dataset.take(slider_input)
    images, labels = list(data)[0]
    images = images.numpy()
    image = images[0]
    fig, ax = plt.subplots()
    ig = plt.imshow(np.squeeze(image[:, :, height_input]), cmap="gray")
    st.pyplot(fig)

    st.text('I found this dataset at...  I decided to work with it because ...')



with modelTraining:
    st.header('Model training')
    if st.button('Train model'):
        with st.spinner("Training ongoing"):
            #Complie model
            
            
            model = get_model()


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

            # Train the model, doing validation at the end of each epoch
            epochs = 5
            model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                shuffle=True,
                verbose=1,
                callbacks=[checkpoint_cb, early_stopping_cb],
            )

            fig, ax = plt.subplots(1, 2, figsize=(30, 5))
            ax = ax.ravel()

            for i, metric in enumerate(["acc", "loss"]):
                ax[i].plot(model.history.history[metric])
                ax[i].plot(model.history.history["val_" + metric])
                ax[i].set_title("Model {}".format(metric))
                ax[i].set_xlabel("epochs")
                ax[i].set_ylabel(metric)
                ax[i].legend(["train", "val"])
            st.pyplot(fig)



with prediction:
    st.header('Predict input image with trained model')
    predict_slider = st.slider('Choose a image from the validation set to predict!', min_value=1, max_value=len(x_val),value=1)
    
    if st.button('Show image'):
        fig, ax = plt.subplots(figsize=(40,3))
        plt.imshow(np.squeeze(x_val[predict_slider,:,:,30]), cmap='gray')
        st.pyplot(fig)

    if st.button('Predict'):
        # Load best weights.
        model = get_model()
        model.load_weights('./3d_image_classification.h5')
        prediction = model.predict(np.expand_dims(x_val[predict_slider], axis=0))[0]
        scores = [1 - prediction[0], prediction[0]]

        class_names = ["normal", "abnormal"]
        for score, name in zip(scores, class_names):
            st.text(
                "This model is %.2f percent confident that CT scan is %s"
                % ((100 * score), name)
            )

    



