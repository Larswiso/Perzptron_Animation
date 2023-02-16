# Externe Bibliotheken
from sklearn import datasets
from sklearn.model_selection import train_test_split

# lokale Anwendung
from perzeptron import Perceptron

# Datensatz generieren
X, y = datasets.make_blobs(n_samples=300, n_features=2, centers=2, cluster_std=1.55, random_state=2)
# Datensatz in Trainingsdaten und Testsdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)   


epochs = 10
perzetron = Perceptron(X_train, y_train, epochs)
perzetron.visualize()

acc_train = perzetron.accuracy(X_train, y_train)
acc_test = perzetron.accuracy(X_test, y_test)

print(f"Perceptrongenauigkeit der Trainingsdaten: {round(acc_train, 2)}")
print(f"Perceptrongenauigkeit der Testsdaten: {round(acc_test, 2)}")