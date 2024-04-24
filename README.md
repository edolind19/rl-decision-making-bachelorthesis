
# Simple Driving Environment mit Q-Learning - Bachelorthesis

Dieses Projekt implementiert eine einfache Fahrumgebung in OpenAI Gym und verwendet Q-Learning, um einen Agenten zu trainieren, der lernen soll, in dieser Umgebung zu navigieren.

## Inhaltsverzeichnis

1. [Beschreibung](#beschreibung)
2. [Voraussetzungen](#voraussetzungen)
3. [Verwendung](#verwendung)
4. [Lizenz](#lizenz)

## Beschreibung

Die Simple Driving Environment ist eine einfache Umgebung, in der ein Agent lernen soll, sich zu bewegen, um ein Ziel zu erreichen, während er Hindernissen ausweicht. Die Umgebung besteht aus einem Gitter mit bestimmten Zuständen und Aktionen. Der Agent verwendet Q-Learning, um eine Q-Tabelle zu aktualisieren, die ihm hilft, die besten Aktionen in jedem Zustand zu wählen.

## Voraussetzungen

Um dieses Projekt auszuführen, benötigen Sie:

- Python 3
- OpenAI Gym
- NumPy
- Matplotlib

## Verwendung

1. Klone das Repository:

   ```bash
   git clone https://github.com/edolind19/rl-decision-making-bachelorthesis.git
   ```

2. Navigiere in das Verzeichnis:

   ```bash
   cd rl-decision-making-bachelorthesis
   ```

3. Führe das Hauptskript aus:

   ```bash
   python main.py
   ```

Das Skript trainiert den Agenten mit Q-Learning und bewertet dann seine Leistung in mehreren Durchläufen. Die Ergebnisse werden in einer CSV-Datei namens `evaluation_results.csv` gespeichert.

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.

