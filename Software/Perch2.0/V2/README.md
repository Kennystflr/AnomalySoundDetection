# Projet : Machine Learning - Audio Anomaly Detection (Intro to Research)

Ce dossier contient l'ensemble du code, des données et des résultats liés au projet de détection d'anomalies audio. L'arborescence a été restructurée pour être claire, professionnelle et prête à être rendue ("propre").

## 📁 Arborescence du projet

Le projet est divisé en 4 dossiers principaux :

* **`src/` (Code source Python)**
  Contient tous les scripts Python exécutables (`.py`). Les codes ont été formatés selon les conventions standards (PEP8 - formateur Black) pour garantir une lecture facile. Les chemins d'accès globaux appellent désormais systématiquement les autres sous-dossiers.
  *Exemples de scripts présents : `Anomaly_detector_forest.py`, `Anomaly_detector_gaussian.py`, `TSNE.py`, etc.*

* **`data/` (Données)**
  Contient les données brutes, les échantillons sonores (`.wav`), les données exportées des modèles (`.npy`, `.json`, `.npz`), et les fichiers d'entrée (`.csv`).

* **`models/` (Modèles Machine Learning)**
  Contient les poids et les modèles pré-entrainés (comme `perch_v2.onnx`) ainsi que les matrices de signatures de références.

* **`results/` (Résultats et livrables)**
  Contient tous les résultats finaux du projet : dashboards HTML intéractifs, graphiques générés (`.png`), et les rapports finaux sous forme de `rapport_anomalies_*.csv`.

## ⚙️ Installation de l'environnement

Pour pouvoir utiliser le code de manière optimale et installer toutes les bibliothèques nécessaires, un fichier `requirements.txt` a été automatiquement généré.

Vous pouvez installer toutes les dépendances via la commande suivante :
```bash
pip install -r requirements.txt
```

*(Si vous vous trouvez dans le dossier parent du projet).*

## 🚀 Execution globale

Les codes ont été configurés de manière à être lancés **depuis le dossier racine du projet**.
Par exemple, pour exécuter l'arbre d'anomalies, lancez :
```bash
python3 src/Anomaly_detector_forest.py
```

## 📦 Rendu
Un fichier `rendu_final.zip` a été généré à la racine. C'est ce fichier qu'il convient de rendre sur la plateforme universitaire. Il inclut toutes vos ressources à l'exception des immenses matrices `.npy` (plusieurs gigaoctets) qui ne rentreraient de toute façon pas sur les espaces de soumission classiques.
