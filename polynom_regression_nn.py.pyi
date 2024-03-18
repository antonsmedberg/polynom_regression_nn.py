polynom_regression_nn.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Definierar konfiguration och hyperparametrar för neurala nätverket och datagenerering
CONFIG = {
    "neurons_layer_1": 64,
    "neurons_layer_2": 64,
    "epochs": 500,
    "batch_size": 32,
    "learning_rate": 0.01,
    "dropout_rate": 0.2,
    "x_start": -1.5,
    "x_end": 1.5,
    "num_points": 1000
}

def generate_data(start, end, num_points):
    """
    Genererar syntetiska data med en polynomfunktion.

    Denna funktion skapar ett dataset baserat på en given polynomfunktion
    för att simulera realistiska datamönster som kan användas för träning av neurala nätverk.
    
    Args:
        start (float): Startvärdet på x-axeln.
        end (float): Slutvärdet på x-axeln.
        num_points (int): Antalet datapunkter som ska genereras.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Returnerar två numpy-arrayer, x och y, representerande datamängden.
    """
    x = np.linspace(start, end, num_points)
    y = -7 * x ** 7 + 4 * x ** 6 + 10 * x ** 5 - 4 * x ** 2 + x - 12
    return x, y

def normalize_data(x, y):
    """
    Normaliserar datan för att förbättra träningseffektiviteten.
    
    Genom att normalisera datan försäkrar vi att neurala nätverket kan träna mer effektivt,
    då det hanterar värden inom ett enhetligt intervall.
    
    Args:
        x (np.ndarray): x-värdena av datamängden.
        y (np.ndarray): y-värdena av datamängden.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, float, float, float, float]: Normaliserade x och y,
        tillsammans med medelvärden och standardavvikelser för originaldatan.
    """
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()
    return (x - x_mean) / x_std, (y - y_mean) / y_std, x_mean, x_std, y_mean, y_std

def build_model(neurons_layer_1, neurons_layer_2, learning_rate, dropout_rate):
    """
    Bygger och returnerar en kompilerad sekventiell TensorFlow-modell.

    Args:
        neurons_layer_1 (int): Antal neuroner i det första dolda lagret.
        neurons_layer_2 (int): Antal neuroner i det andra dolda lagret.
        learning_rate (float): Inlärningshastigheten för optimeraren.
        dropout_rate (float): Dropout rate för att förhindra overfitting.
    
    Returns:
        tf.keras.Model: Den kompilerade modellen.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons_layer_1, input_shape=(1,), activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(neurons_layer_2, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def create_learning_rate_scheduler(initial_learning_rate):
    """
    Skapar en scheduler för dynamisk anpassning av inlärningshastigheten under träning.

    Args:
        initial_learning_rate (float): Startvärdet på inlärningshastigheten.

    Returns:
        tf.keras.callbacks.Callback: Callback-funktion för dynamisk anpassning av inlärningshastigheten.
    """
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    return tf.keras.callbacks.LearningRateScheduler(scheduler)

def plot_loss(history):
    """
    Plottar tränings- och valideringsförlust över tiden för att visualisera modellens lärande.
    
    Args:
        history (tf.keras.callbacks.History): Träningshistoriken från modell.fit.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Träningsförlust')
    plt.plot(history.history['val_loss'], label='Valideringsförlust')
    plt.title('Förlust Över Tid')
    plt.xlabel('Epok')
    plt.ylabel('Förlust')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_model(config):
    """
    Kör hela processen från dataförberedelse till modellträning och utvärdering.

    Args:
        config (dict): En konfigurationsdictionary som innehåller modellens hyperparametrar och datainställningar.
    """
    try:
        x, y = generate_data(config["x_start"], config["x_end"], config["num_points"])
        x_normalized, y_normalized, _, _, _, _ = normalize_data(x, y)

        model = build_model(config["neurons_layer_1"], config["neurons_layer_2"], config["learning_rate"], config["dropout_rate"])
        lr_scheduler = create_learning_rate_scheduler(config["learning_rate"])

        history = model.fit(x_normalized, y_normalized, batch_size=config["batch_size"], epochs=config["epochs"], 
                            validation_split=0.2, callbacks=[lr_scheduler])

        plot_loss(history)
        model.save('/mnt/data/polynomial_regression_model.h5')
    except Exception as e:
        print(f"Ett fel uppstod under körningen: {e}")

if __name__ == "__main__":
    run_model(CONFIG)

