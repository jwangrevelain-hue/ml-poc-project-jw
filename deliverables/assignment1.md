# Assignment 1 — Cadrage du projet

**Auteur :** Joseph Wangrevelain
**Dataset :** Craigslist Cars and Trucks Data (Kaggle)
**Date :** 2026-04-29

---

## 1. Description du projet et objectif business

Le marché des véhicules d'occasion est particulièrement dynamique, mais l'asymétrie d'information entre acheteurs et vendeurs rend souvent la fixation du juste prix difficile. L'objectif de ce projet est de développer un **modèle d'estimation automatique du prix d'un véhicule d'occasion** en se basant sur ses caractéristiques techniques et son état.

### Objectif business

Fournir un outil d'aide à la décision fiable :

- **Pour les vendeurs** : estimer le prix de vente optimal pour vendre rapidement sans brader le véhicule.
- **Pour les acheteurs** : identifier rapidement les "bonnes affaires" (véhicules sous-cotés) et éviter les arnaques.

---

## 2. Définition du problème et contexte ML

| Élément | Choix |
|---|---|
| Type de tâche | Apprentissage supervisé (*Supervised Learning*) |
| Type de problème | **Régression** |
| Variable cible | `price` (prix du véhicule, en dollars) |

Nous cherchons à prédire une valeur continue (le prix en dollars) à partir d'un ensemble de variables explicatives (features).

---

## 3. Description du dataset choisi

Le jeu de données utilisé est le **Craigslist Cars and Trucks Data**, disponible sur Kaggle. Il s'agit d'un scraping massif des annonces de véhicules d'occasion publiées sur le site américain Craigslist.

- **Taille d'origine** : plus de **420 000 lignes** et **26 colonnes**.
  *Note : un sous-échantillonnage pourra être envisagé si les temps de calcul sont trop longs.*
- **Variable cible (target)** : `price` (prix du véhicule en dollars).

### 3.1 Justification du choix du dataset

Ce dataset a été retenu pour plusieurs raisons :

- **Praticité d'accès** : disponible directement sur Kaggle, sans contraintes d'authentification ou de scraping en direct.
- **Volume et richesse** : c'est le dataset le plus fourni sur le sujet des véhicules d'occasion, ce qui ouvre de nombreuses possibilités de **feature engineering** (variables techniques, géographiques, temporelles).
- **Origine** : il résulte d'un scraping régulier d'un site de vente d'occasion réel, ce qui le rend représentatif du marché.
- **Flexibilité** : la diversité des features permet une plus grande latitude dans le choix des transformations et des modèles à comparer.

### 3.2 Description des features disponibles

Les caractéristiques peuvent être divisées en plusieurs catégories :

| Catégorie | Features |
|---|---|
| **Numériques** | `year` (année de fabrication), `odometer` (kilométrage en miles) |
| **Catégorielles (techniques)** | `manufacturer` (marque), `model` (modèle), `condition` (état général), `cylinders` (nombre de cylindres), `fuel` (type de carburant), `transmission` (boîte de vitesse), `drive` (type de transmission/roues motrices), `size` (taille du véhicule), `type` (catégorie : SUV, berline, etc.), `paint_color` (couleur) |
| **Géographiques** | `state` (état américain), `region` (région), `lat` (latitude), `long` (longitude) |
| **Temporelles** | `posting_date` (date de publication de l'annonce) |
| **Texte / identifiants (à traiter ou exclure)** | `id`, `url`, `region_url`, `image_url`, `description` (texte libre), `VIN` (numéro de série) |

---

## 4. Premières analyses exploratoires (EDA) envisagées

Pour comprendre la structure des données avant modélisation, les axes d'exploration suivants seront privilégiés :

### 4.1 Analyse de la target (`price`)
- Étude de la distribution des prix pour identifier la présence (très probable) de **valeurs extrêmes (outliers)**, comme des prix aberrants à `0 $` ou `1 000 000 $`.

### 4.2 Analyse des valeurs manquantes
- Visualisation des taux de remplissage pour chaque colonne, particulièrement pour `condition`, `size` et `cylinders` qui sont souvent incomplètes sur Craigslist.

### 4.3 Corrélations bivariées
- Impact du kilométrage (`odometer`) sur le prix — relation inversement proportionnelle attendue.
- Impact de l'âge du véhicule (calculé à partir de `year`) sur le prix.
- Différences de prix médian selon les principales marques (`manufacturer`).

---

## 5. Métriques d'évaluation envisagées

Pour évaluer les performances de notre modèle de régression, nous utiliserons principalement :

| Métrique | Justification |
|---|---|
| **MAE** (*Mean Absolute Error*) | Très interprétable d'un point de vue business : elle donne l'erreur moyenne de prédiction en dollars. Plus robuste aux valeurs extrêmes que la MSE. |
| **RMSE** (*Root Mean Squared Error*) | Utile pour pénaliser plus fortement les grosses erreurs de prédiction (ex. une erreur de 10 000 $ sur une voiture sera fortement sanctionnée). |

---

## 6. Hypothèses, risques et limites identifiés

| Risque / limite | Description |
|---|---|
| **Bruit dans la variable cible** | Les prix affichés sur Craigslist sont des *prix demandés* par les vendeurs, et non les prix de vente finaux après négociation. De plus, de fausses annonces à 1 $ existent pour attirer l'attention. |
| **Qualité des données** | Le dataset contient un grand nombre de valeurs manquantes (`NaN`) qu'il faudra gérer (imputation intelligente ou suppression). |
| **Biais géographique** | Les données sont exclusivement américaines. Les marques (beaucoup de Ford/Chevrolet), les types de véhicules (beaucoup de pick-ups) et les prix ne sont pas représentatifs du marché européen. |
| **Volume / cardinalité** | Le grand nombre de catégories pour la variable `model` risque de créer une matrice très creuse lors de l'encodage (risque de surapprentissage ou de lenteur d'exécution). |

---

## 7. Perspectives d'enrichissement

Pour aller plus loin, l'utilisation d'un **second dataset complémentaire** est envisagée : un dataset listant le **prix des voitures neuves**.

**Intérêt** : croiser les deux sources permettrait de **mieux détecter et exclure les outliers** dans le dataset Craigslist, en s'appuyant sur la règle simple suivante :

> Une voiture d'occasion ne devrait jamais être affichée à un prix supérieur à celui de la même voiture en neuf.

**Source candidate** : [Car Features and MSRP — CooperUnion (Kaggle)](https://www.kaggle.com/datasets/CooperUnion/cardataset).

Cet enrichissement n'est pas requis pour le POC initial mais constitue une piste d'amélioration pour les itérations suivantes du projet.