import pandas as pd

# Daten laden
data = pd.read_csv('csv/evaluation_results.csv')

# Annahme: Die Daten sind abwechselnd, also jede zweite Zeile
data_with_algo = data.iloc[1::2]  # Zeilen mit ungeraden Indizes verwenden den Algorithmus
data_without_algo = data.iloc[::2]  # Zeilen mit geraden Indizes verwenden keinen Algorithmus

# Mittelwerte berechnen
mean_with_algo = data_with_algo.mean()
mean_without_algo = data_without_algo.mean()


with open('csv/mean_results.csv', 'w') as file:
    file.write("Mittelwerte mit Algorithmus:\n")
    file.write(mean_with_algo.to_string())
    file.write("\n\nMittelwerte ohne Algorithmus:\n")
    file.write(mean_without_algo.to_string())
# Ausgabe der Mittelwerte
print("Mittelwerte mit Algorithmus:")
print(mean_with_algo)
print("\nMittelwerte ohne Algorithmus:")
print(mean_without_algo)
