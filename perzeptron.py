# Standardbibliotheken 
import datetime
import os
from pathlib import Path

# externe Bibliotheken
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Perceptron:
    def __init__(self, X_train, y_train, epochs):
        # Initialisiere die Gewichte mit zufälligen Werten, den Bias-Term mit 0.
        self.weights = np.random.rand(X_train.shape[1])
        self.bias = 0
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs

    def predict(self, inputs):
        # Berechne die gewichtete Summe der Eingaben und füge den Bias-Term hinzu.
        weighted_sum = np.dot(inputs, self.weights) + self.bias

        # Verwende eine Aktivierungsfunktion, um die gewichtete Summe in eine binäre Vorhersage (0 oder 1) zu transformieren.
        return 1 if weighted_sum >= 0 else 0
    
    def learn(self, training_inputs, labels, epoch,learning_rate=0.1):
        for inputs, label in zip(training_inputs, labels):
            # Berechne die Vorhersage des Perzeptrons für die aktuellen Eingaben.
            prediction = self.predict(inputs)
    
            if prediction != label:
                    self.fehler[epoch] += 1
       

            # Berechne den Fehler zwischen der tatsächlichen Ausgabe und der erwarteten Ausgabe
            error = label - prediction

            # Berechne die Änderung der Gewichte
            delta_weights = learning_rate * error * inputs
            # Anpassen der Gewichte
            self.weights = self.weights + delta_weights 

            # Berechne die Änderung des Bias
            delta_bias = learning_rate * error
            # Anpassen des Bias
            self.bias = self.bias + delta_bias 

    def update(self, epoch):
        
        self.fehler = np.zeros(self.epochs)
        self.learn(self.X_train, self.y_train, epoch)
        # Berechne die Trennlinie
        w1 = self.weights[0]
        w2 = self.weights[1]
        b = self.bias
        x_line = np.array([min(self.X_train[:, 0]), max(self.X_train[:, 0])])
        y_line = (-b - w1 * x_line) / w2

        ax.cla()
        # Visualisiere die Eingabedaten
        ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap='coolwarm', label="Klasse 0")

        # add text outside the plot
        
        
        ax.set_title(f"Trainingsdaten | Perzeptrongenauigkeit: {round(self.accuracy(self.X_train, self.y_train), 2)}")
        ax.plot(x_line, y_line, 'y', label="Trennlinie")


        
    def visualize(self, save_Animation=True):
        
        ani = FuncAnimation(fig, self.update, frames=self.epochs, interval=200)
        
        if save_Animation:
            current_directory = Path(__file__).parent.absolute()

            output_directory = os.path.join(current_directory, "output")
            filename = "perzeprtron" + datetime.datetime.now().strftime("%H_%M_%S") + ".gif"
            ani.save(os.path.join(output_directory, filename), writer='pillow')
            ani = None

        plt.show()

    def accuracy(self, X_arry, y_array):
        pred = [self.predict(x) for x in X_arry]
        accuracy = np.sum(y_array == pred) / len(y_array)
        return accuracy   

fig, ax = plt.subplots()
