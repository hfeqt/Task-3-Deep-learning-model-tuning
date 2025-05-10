import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import os


os.makedirs("plots", exist_ok=True)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0


configs = [
    {"neurons": 64, "batch_size": 32, "epochs": 50},
    {"neurons": 256, "batch_size": 64, "epochs": 50},
    {"neurons": 512, "batch_size": 64, "epochs": 100},
]

for config in configs:
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(config["neurons"], activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=config["epochs"],
                        batch_size=config["batch_size"],
                        verbose=0)

    
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Loss - {config}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"plots/loss_{config['neurons']}_{config['batch_size']}_{config['epochs']}.png")

  
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Accuracy - {config}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"plots/acc_{config['neurons']}_{config['batch_size']}_{config['epochs']}.png")
