from distro import name
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    LeakyReLU,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import pickle


def get_data_generators(train_path, test_path):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255.0,
        horizontal_flip=True,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    train = datagen.flow_from_directory(
        train_path,
        target_size=(128, 128),
        class_mode="sparse",
        seed=1,
        color_mode="grayscale",
        batch_size=128,
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
    test = test_datagen.flow_from_directory(
        test_path,
        target_size=(128, 128),
        class_mode="sparse",
        seed=1,
        color_mode="grayscale",
        batch_size=128,
    )
    return train, test


def build_model():
    model = Sequential(name="DCNN")
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(5, 5),
            input_shape=(128, 128, 1),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_1",
        )
    )
    model.add(BatchNormalization(name="batchnorm_1"))
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(5, 5),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_2",
        )
    )
    model.add(BatchNormalization(name="batchnorm_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_1"))
    model.add(Dropout(0.4, name="dropout_1"))
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_3",
        )
    )
    model.add(BatchNormalization(name="batchnorm_3"))
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_4",
        )
    )
    model.add(BatchNormalization(name="batchnorm_4"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_2"))
    model.add(Dropout(0.4, name="dropout_2"))
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_5",
        )
    )
    model.add(BatchNormalization(name="batchnorm_5"))
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_6",
        )
    )
    model.add(BatchNormalization(name="batchnorm_6"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_3"))
    model.add(Dropout(0.5, name="dropout_3"))
    model.add(Flatten(name="flatten"))
    model.add(
        Dense(128, activation="elu", kernel_initializer="he_normal", name="dense_1")
    )
    model.add(BatchNormalization(name="batchnorm_7"))
    model.add(Dropout(0.6, name="dropout_4"))
    model.add(Dense(len(train.class_indices), activation="softmax", name="out_layer"))
    return model


def train_model(model, train, test):
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.00005,
        patience=11,
        verbose=1,
        restore_best_weights=True,
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train,
        epochs=150,
        validation_data=test,
        callbacks=[early_stopping, lr_scheduler],
    )
    return history


def plot_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(history.history["loss"])
    axs[0].plot(history.history["val_loss"])
    axs[0].set_title("Model Loss")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(["Train", "Validation"], loc="upper right")

    axs[1].plot(history.history["accuracy"])
    axs[1].plot(history.history["val_accuracy"])
    axs[1].set_title("Model Accuracy")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(["Train", "Validation"], loc="lower right")
    plt.show()


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_path = "dataset/train"
    test_path = "dataset/test"
    train, test = get_data_generators(train_path, test_path)
    model = build_model()
    history = train_model(model, train, test)
    save_model(model, "model.pkl")
    plot_history(history)
