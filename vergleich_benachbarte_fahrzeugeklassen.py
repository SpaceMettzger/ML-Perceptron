import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('AutodatenbankAllEntries.csv', header=0)

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

    print("=====================================")
    print(f"Verleich Klassen {i} und {i+1}")
    print("Genauigkeit trainierter Daten, Ursprungsdaten standard skaliert:")
    acc_standard = accuracy_score(y_pred_scaled_standard, y_train)
    print("Accuracy: ", acc_standard)
    print("Genauigkeit trainierter Daten, Ursprungsdaten normal skaliert:")
    acc_norm = accuracy_score(y_pred_scaled_norm, y_train)
    print("Accuracy: ", acc_norm)
    print("Genauigkeit trainierter Daten, Ursprungsdaten robust skaliert:")
    acc_robust = accuracy_score(y_pred_scaled_robust, y_train)
    print("Accuracy: ", acc_robust)
