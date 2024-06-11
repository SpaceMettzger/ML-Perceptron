"""
Training von neuronalem Netz (einlagiges Perzeptron)
Zwecks Training werden jeweils die ersten 30 Samples (vom Kopf der Datei beginnend) je Fahrzeugklasse verwendet.
Der Einfachheit halber beschränken wir uns dieses Mal auf nur zwei Merkmale (Grundpreis in € und Leistung in kW).
"""

import pandas as pd
import numpy as np
#  from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score


data = pd.read_csv('AutodatenbankAllEntries.csv', header=0)

data['Auto'] = data['Marke'] + ' ' + data['Modell'] + ' ' + data['Grundpreis'].astype(str)

cars_to_exclude = [
    "smart fortwo Coupee 1.0 11105",
    "SEAT Ibiza 1.0 MPI 12490",
    "Subaru XV 1.6i 22980",
    "Skoda Octavia Combi RS 31590"
]

exclude_indexes = data.index[data['Auto'].isin(cars_to_exclude)].tolist()
excluded_data = data.loc[exclude_indexes]
data_remaining = data.drop(index=exclude_indexes)

print(excluded_data)

data_remaining.reset_index(drop=True, inplace=True)

df_train = pd.concat([data_remaining[data_remaining['Fahrzeugklasse'] == i].head(30) for i in range(1, 7)])
df_test = excluded_data

x_train = df_train[['Grundpreis', 'Leistung_kW']]
y_train = df_train['Fahrzeugklasse']

x_test = df_test[['Grundpreis', 'Leistung_kW']]
y_test = df_test['Fahrzeugklasse']


"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()  # Standardisierung
X_train_scaled_standard = scaler.fit_transform(X_train)
X_test_scaled_standard = scaler.transform(X_test)

scaler2 = MinMaxScaler()  # Normierung
X_train_scaled_norm = scaler2.fit_transform(X_train)
X_test_scaled_norm = scaler2.transform(X_test)

scaler3 = RobustScaler()  # Robust
X_train_scaled_robust = scaler3.fit_transform(X_train)
X_test_scaled_robust = scaler3.transform(X_test)
"""

scaler = StandardScaler()  # Standardisierung
x_train_scaled_standard = scaler.fit_transform(x_train)
x_test_scaled_standard = scaler.transform(x_test)

scaler2 = MinMaxScaler()  # Normierung
x_train_scaled_norm = scaler2.fit_transform(x_train)
x_test_scaled_norm = scaler2.transform(x_test)

scaler3 = RobustScaler()  # Robust
x_train_scaled_robust = scaler3.fit_transform(x_train)
x_test_scaled_robust = scaler3.transform(x_test)


max_iter = 100  # Dieser Wert gibt an, wie viele Iterationen durchlaufen werden, in denen die Gewichte angepasst werden.
# Alle Werte kleiner als 24 geben Warnungen aus

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

print("Testdaten:")
y_test_array = np.array(y_test.to_list())
print(y_test_array)
print("=====================================")
print("Trainierte daten, Ursprungsdaten standard skaliert:")
acc_standard = accuracy_score(y_pred_scaled_standard, y_test_array)
print(y_pred_scaled_standard, ", Accuracy: ", acc_standard)
print("Trainierte daten, Ursprungsdaten normal skaliert:")
acc_norm = accuracy_score(y_pred_scaled_norm, y_test_array)
print(y_pred_scaled_norm, ", Accuracy: ", acc_norm)
print("Trainierte daten, Ursprungsdaten robust skaliert:")
acc_robust = accuracy_score(y_pred_scaled_robust, y_test_array)
print(y_pred_scaled_robust, ", Accuracy: ", acc_robust)

