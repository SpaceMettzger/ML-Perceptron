"""
Training von neuronalem Netz (einlagiges Perzeptron)
Zwecks Training werden jeweils die ersten 30 Samples (vom Kopf der Datei beginnend) je Fahrzeugklasse verwendet.
Der Einfachheit halber beschränken wir uns dieses Mal auf nur zwei Merkmale (Grundpreis in € und Leistung in kW).
"""

import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def print_scales(standard, normiert, robust_skaliert, data):
    print("Wertebereiche der Attribute:")
    print("Standardisiert:")
    data_df = pd.DataFrame(standard, columns=data.columns)
    average_1 = data_df['Grundpreis'].sum() / data_df['Grundpreis'].count()
    print("\tGrundpreis", round(data_df['Grundpreis'].min(), 4), "bis",
          round(data_df['Grundpreis'].max(), 4),
          ", Durchschnitt: ", average_1)
    print("\tLeistung_kW", round(data_df['Leistung_kW'].min(), 4), "bis ",
          round(data_df['Leistung_kW'].max(), 4), "\n")

    print("Normiert:")
    x_train_scaled_norm_df = pd.DataFrame(normiert, columns=data.columns)
    average_2 = x_train_scaled_norm_df['Grundpreis'].sum() / x_train_scaled_norm_df['Grundpreis'].count()
    print("\tGrundpreis", x_train_scaled_norm_df['Grundpreis'].min(), "bis", x_train_scaled_norm_df['Grundpreis'].max(),
          ", Durchschnitt: ", average_2)
    print("\tLeistung_kW", round(x_train_scaled_norm_df['Leistung_kW'].min(), 4), "bis ",
          round(x_train_scaled_norm_df['Leistung_kW'].max(), 4), "\n")

    print("Robust Skaliert:")
    x_train_scaled_robust_df = pd.DataFrame(robust_skaliert, columns=data.columns)
    average_3 = x_train_scaled_robust_df['Grundpreis'].sum() / x_train_scaled_robust_df['Grundpreis'].count()
    print("\tGrundpreis", round(x_train_scaled_robust_df['Grundpreis'].min(), 4), "bis ",
          round(x_train_scaled_robust_df['Grundpreis'].max(), 4),
          ", Durchschnitt: ", average_3)
    print("\tLeistung_kW", round(x_train_scaled_robust_df['Leistung_kW'].min(), 4), "bis ",
          round(x_train_scaled_robust_df['Leistung_kW'].max(), 4), "\n")


def calculate_average_based_on_sums(perceptron, data, scaler):
    weights = perceptron.coef_
    intercept = perceptron.intercept_
    """
    print("Alle Koeffizienten:\n", weights)
    print("\n\nGewichtung für Perzeptron:")
    for i, col in enumerate(x_train.columns):
        print(f"{col}: {weights[0][i]}")
    print(f"Intercept: {intercept[0]}\n")
    """

    scaled_data = scaler.transform(data)
    price_1 = scaled_data[0][0]
    kw_1 = scaled_data[0][1]
    price_2 = scaled_data[1][0]
    kw_2 = scaled_data[1][1]
    price_3 = scaled_data[2][0]
    kw_3 = scaled_data[2][1]
    price_4 = scaled_data[3][0]
    kw_4 = scaled_data[3][1]

    anteile_klassen = {}

    for i in range(0, weights.shape[0]):
        sum_1 = abs(weights[i][0]) * price_1 + abs(weights[i][1]) * kw_1
        sum_2 = abs(weights[i][0]) * price_2 + abs(weights[i][1]) * kw_2
        sum_3 = abs(weights[i][0]) * price_3 + abs(weights[i][1]) * kw_3
        sum_4 = abs(weights[i][0]) * price_4 + abs(weights[i][1]) * kw_4

        anteil = abs(weights[i][0]) * (price_1 + price_2 + price_3 + price_4) / (sum_1 + sum_2 + sum_3 + sum_4)
        anteile_klassen[i] = anteil

        print(f"Anteil Grundpreis an Klasse {i + 1}: {anteil}")

    print("Anteil Grundpreis an allen Klassen: ",
          sum([value for value in anteile_klassen.values()]) / len(anteile_klassen))


def calculate_average_based_on_weights(perceptron, data, scaler):
    scaled_input = scaler.transform(data)
    weights = perceptron.coef_
    intercept = perceptron.intercept_
    sums = []
    for input in scaled_input:
        price, kwh = input[0], input[1]
        sub_sums = []
        for i in range(0, weights.shape[0]):
            sub_sums.append(weights[i][0] * price + weights[i][1] * kwh + intercept[i])
        sums.append(sub_sums)

    print([item.index(max(item)) + 1 for item in sums])


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

data_remaining.reset_index(drop=True, inplace=True)

df_train = pd.concat([data_remaining[data_remaining['Fahrzeugklasse'] == i].head(30) for i in range(1, 7)])
df_test = excluded_data

x_train = df_train[['Grundpreis', 'Leistung_kW']]
y_train = df_train['Fahrzeugklasse']

x_test = df_test[['Grundpreis', 'Leistung_kW']]
y_test = df_test['Fahrzeugklasse']

scaler1 = StandardScaler()  # Standardisierung
x_train_scaled_standard = scaler1.fit_transform(x_train)
x_test_scaled_standard = scaler1.transform(x_test)

scaler2 = MinMaxScaler()  # Normierung
x_train_scaled_norm = scaler2.fit_transform(x_train)
x_test_scaled_norm = scaler2.transform(x_test)

scaler3 = RobustScaler()  # Robust
x_train_scaled_robust = scaler3.fit_transform(x_train)
x_test_scaled_robust = scaler3.transform(x_test)


max_iter = 100  # Dieser Wert gibt an, wie viele Iterationen durchlaufen werden, in denen die Gewichte angepasst werden.
# Alle Werte kleiner als 24 geben Warnungen aus (abhängig vom Zufallswert)

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


print_scales(x_train_scaled_standard, x_train_scaled_norm, x_train_scaled_robust, x_train)

print("Perceptron 1:")
calculate_average_based_on_sums(perceptron1, x_test, scaler1)
print("\nPerceptron 2:")
calculate_average_based_on_sums(perceptron2, x_test, scaler2)
print("\nPerceptron 3:")
calculate_average_based_on_sums(perceptron3, x_test, scaler3)

print("\nVersuch der manuellen Klassifizierung mit den Gewichten der Perzeptronen:")
calculate_average_based_on_weights(perceptron1, x_test, scaler1)
calculate_average_based_on_weights(perceptron2, x_test, scaler2)
calculate_average_based_on_weights(perceptron3, x_test, scaler3)

