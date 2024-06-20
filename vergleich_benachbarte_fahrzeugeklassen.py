import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('AutodatenbankAllEntries.csv', header=0)

data_output = {}

for i in range(1, 6):
    df = pd.concat([data[data['Fahrzeugklasse'] == j] for j in range(i, i+2)])
    x_train = df[['Grundpreis', 'Leistung_kW']]
    y_train = df['Fahrzeugklasse']

    scaler = StandardScaler()  # Standardisierung
    x_train_scaled_standard = scaler.fit_transform(x_train)
    x_test_scaled_standard = scaler.transform(x_train)

    scaler2 = MinMaxScaler()  # Normierung
    x_train_scaled_norm = scaler2.fit_transform(x_train)
    x_test_scaled_norm = scaler2.transform(x_train)

    scaler3 = RobustScaler()  # Robust
    x_train_scaled_robust = scaler3.fit_transform(x_train)
    x_test_scaled_robust = scaler3.transform(x_train)

    max_iter = 1000
    rand_state = 42

    perceptron1 = Perceptron(max_iter=max_iter, random_state=rand_state)
    perceptron1.fit(x_train_scaled_standard, y_train)

    perceptron2 = Perceptron(max_iter=max_iter, random_state=rand_state)
    perceptron2.fit(x_train_scaled_norm, y_train)

    perceptron3 = Perceptron(max_iter=max_iter, random_state=rand_state)
    perceptron3.fit(x_train_scaled_robust, y_train)

    y_pred_scaled_standard = perceptron1.predict(x_test_scaled_standard)
    y_pred_scaled_norm = perceptron1.predict(x_test_scaled_norm)
    y_pred_scaled_robust = perceptron1.predict(x_test_scaled_robust)

    acc_standard = accuracy_score(y_pred_scaled_standard, y_train)
    acc_norm = accuracy_score(y_pred_scaled_norm, y_train)
    acc_robust = accuracy_score(y_pred_scaled_robust, y_train)

    output_label = f"{i} und {i+1}"
    data_output[output_label] = {
        "standardisiert": acc_standard,
        "normiert": acc_norm,
        "robust_skaliert": acc_robust
    }

print("=====================================")
print("Vergleich mit standardisierten Daten:")
for key in data_output.keys():
    print(f"{key}: {data_output[key]['standardisiert']}")
print("=====================================")
print("Vergleich mit normierten Daten:")
for key in data_output.keys():
    print(f"{key}: {data_output[key]['normiert']}")
print("=====================================")
print("Vergleich mit robust skalierten Daten:")
for key in data_output.keys():
    print(f"{key}: {data_output[key]['robust_skaliert']}")

max_accuracy = 0
classes = ""
scaling = ""
for key in data_output.keys():
    if data_output[key]['standardisiert'] > max_accuracy:
        max_accuracy = data_output[key]['standardisiert']
        classes = key
        scaling = "standardisiert"
    if data_output[key]['normiert'] > max_accuracy:
        max_accuracy = data_output[key]['normiert']
        classes = key
        scaling = "normiert"
    if data_output[key]['robust_skaliert'] > max_accuracy:
        max_accuracy = data_output[key]['robust_skaliert']
        classes = key
        scaling = "robust_skaliert"

print("\n\n=====================================")
print(f"Maximale Genauigkeit von {max_accuracy} wurde für die Klassen {classes} mit skalierung '{scaling}' erreicht")
