.. catemotionrecognition documentation master file, created by
   sphinx-quickstart on Thu Dec  5 23:02:11 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction about catemotionrecognition Project
=================================================

Table des Matières
==================
1. Résumé
2. Introduction
3. Reconnaissance des émotions du chat par la voix
   - Collecte des données
   - Prétraitement des données
   - Construction des modèles
   - Évaluation des performances
   - Collecte des données
   - Prétraitement des données
   - Construction des modèles
   - Évaluation des performances
5. Reconnaissance de l’état de santé du chat par l’image
   - Collecte des données
   - Prétraitement des données
   - Construction des modèles
   - Évaluation des performances
6. Combinaison des modèles
7. Déploiement
8. Conclusion
---

## Résumé
Notre projet consiste à élaborer un modèle de machine learning capable de diagnostiquer l’état et l’humeur de l’animal domestique, le chat, en temps réel. On a conçu 3 modèles. Le 1er permet de générer une prédiction sur l’humeur à partir des images capturées, le 2e consiste à détecter aussi l’humeur, mais cette fois à partir des audios comme input. Le 3e permet de faire une prédiction sur l’état du chat à partir des images, en identifiant si son état est normal ou s’il est malade. Ensuite, on a combiné ces modèles pour générer une prédiction sur des vidéos réelles captées du chat. Enfin, on a fait le déploiement sur Streamlit et une application bureau.
---

## Introduction

Avec l’avancée rapide des technologies en intelligence artificielle et en machine learning, de nouvelles opportunités se présentent pour améliorer le bien-être des animaux domestiques. Notre projet s’inscrit dans cette démarche en visant à développer une solution innovante capable de diagnostiquer en temps réel l’état de santé et l’humeur d’un animal domestique, en particulier le chat. Ce travail repose sur l’analyse de données multimodales telles que les images, les audios, et les vidéos pour fournir des informations précises et utiles aux propriétaires d’animaux.
À travers ce projet, nous avons conçu et combiné plusieurs modèles de machine learning pour détecter non seulement les émotions du chat, mais aussi son état de santé, et avons déployé la solution sur des plateformes accessibles comme Streamlit et une application de bureau. L’objectif final est d’offrir un outil pratique et efficace permettant une interaction enrichie entre les propriétaires et leurs animaux tout en veillant à leur bien-être.

---

## Reconnaissance des émotions du chat par la voix

### Collecte des données

Description des étapes pour collecter les données vocales des chats.

### Prétraitement des données

Techniques utilisées pour nettoyer et préparer les données vocales pour l’entraînement.

### Construction des modèles

Architecture et algorithmes choisis pour le modèle de reconnaissance des émotions basé sur la voix.

### Évaluation des performances

Mesures et résultats des tests effectués sur le modèle.

---

## Reconnaissance des émotions du chat par l’image

### Collecte des données

Description des étapes pour collecter les images nécessaires.

### Prétraitement des données

Techniques utilisées pour nettoyer et préparer les images pour l’entraînement.

### Construction des modèles

Architecture et algorithmes choisis pour le modèle de reconnaissance des émotions basé sur l’image.

### Évaluation des performances

Mesures et résultats des tests effectués sur le modèle.

---

## Reconnaissance de l’état de santé du chat par l’image

### Collecte des données

Description des étapes pour collecter des données sur l’état de santé des chats.

### Prétraitement des données

Techniques utilisées pour préparer les données visuelles pour l’entraînement.

### Construction des modèles

Détails des modèles utilisés pour prédire l’état de santé.

### Évaluation des performances

Analyse des performances du modèle.

---

## Combinaison des modèles

Stratégies utilisées pour combiner les modèles d’analyse vocale et visuelle afin de fournir des prédictions plus complètes.

---

## Déploiement

Étapes pour déployer la solution sur Streamlit et l’application de bureau, avec les outils et frameworks utilisés.

---

## Conclusion

Résumé des résultats obtenus, des défis rencontrés et des perspectives d’avenir pour le projet.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
.. toctree::
   :maxdepth: 2
   :caption: Table des Matières
   résumé
   introduction
   reconnaissance_voix
   reconnaissance_image
   reconnaissance_santé
   combinaison
   déploiement
   conclusion
