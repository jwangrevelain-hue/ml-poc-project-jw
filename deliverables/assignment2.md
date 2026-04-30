# Assignment 2 — Préparation et nettoyage des données

**Auteur :** Joseph Wangrevelain
**Dataset principal :** Craigslist Cars and Trucks Data (Kaggle)
**Dataset complémentaire :** Car Features and MSRP — Cooper Union (Kaggle)
**Date :** 2026-04-29

---

## 1. Vue d'ensemble

Cet assignment décrit la **pipeline de préparation des données** appliquée au dataset Craigslist (~426 000 annonces de véhicules d'occasion) pour produire un dataset propre, encodé et prêt à être consommé par les modèles.

Le pipeline complet est implémenté dans [`notebooks/02_cleaning.ipynb`](../notebooks/02_cleaning.ipynb), exécutable de bout en bout. Il produit un fichier `data/vehicles_processed.parquet` que le module [`src/data.py`](../src/data.py) lit pour fournir les splits `(X_train, X_test, y_train, y_test)`.

**Sources d'entrée**

| Fichier | Description | Taille |
|---|---|---|
| `data/vehicles.csv` | Annonces Craigslist (target : `price`) | ~1.3 GB, 426k lignes, 26 cols |
| `data/new_cars_msrp.csv` | Prix de vente neufs (MSRP) — Cooper Union | ~1.4 MB, 11k lignes, 16 cols |

**Sortie**

| Fichier | Contenu |
|---|---|
| `data/vehicles_processed.parquet` | Données nettoyées + encodées + scalées + colonne `_split` (`"train"` / `"test"`) + target `price` |

> ⚠️ Les CSV sources et le parquet processé sont **exclus du repo** (`.gitignore`) car trop volumineux pour GitHub (limite 100 MB / fichier).

---

## 2. Étapes de nettoyage des données

### 2.1 Drop des colonnes inutiles

À la lecture, on supprime 5 colonnes qui n'apportent aucun signal prédictif et alourdissent la mémoire :

| Colonne | Raison du drop |
|---|---|
| `id` | Identifiant unique, aucune information |
| `url`, `region_url`, `image_url` | URLs, non exploitables tels quels |
| `VIN` | Numéro de série, identifiant unique |

**À conserver (pour le moment) : `description`** — texte libre. On l'exploite par regex pour extraire 3 features texte (cf. § 3) avant de la supprimer en fin de feature engineering.

### 2.2 Filtres de cohérence basique

Avant tout traitement avancé, on applique des filtres "de bon sens" :

| Filtre | Justification |
|---|---|
| `500 ≤ price ≤ 250 000` | Élimine les annonces à 0 $ / 1 $ (clickbait, "contactez-moi") et les valeurs absurdes (> 250k $) |
| `1980 ≤ year ≤ 2022` | Avant 1980 → marché de collection (bruit énorme) ; après 2022 → impossible (scraping en 2021) |
| `manufacturer` non-NaN | Sans la marque, impossible de matcher avec MSRP ni de modéliser correctement |
| `0 ≤ odometer ≤ 500 000` (ou NaN) | Au-delà de 500k miles : majoritairement des typos |

### 2.3 Détection d'outliers via le dataset MSRP (étape clé)

C'est l'apport principal de ce 2ᵉ rendu : utiliser le **dataset des prix neufs** pour détecter les annonces aberrantes.

**Règle métier (stricte)** :
> Une voiture d'occasion ne devrait **jamais** être affichée à un prix supérieur au MSRP (Manufacturer's Suggested Retail Price) de la même marque la même année.

**Implémentation**

1. Normalisation des noms de marque dans les deux datasets (`lowercase`, `strip`, normalisation des espaces) → permet le matching robuste (`BMW` ⇆ `bmw`, `Mercedes-Benz` ⇆ `mercedes-benz`).
2. Construction d'une table de référence `(marque, année) → MSRP_max` (max sur toutes les déclinaisons d'un modèle dans la même année/marque).
3. Jointure left de cette table sur le dataset Craigslist.
4. Filtrage strict : on retire **toutes** les lignes où `price > MSRP_max`.

**Pourquoi `MSRP_max` (et pas `MSRP_mean`)** ?

On prend le **maximum** sur toutes les déclinaisons d'un modèle pour la même marque/année, afin de ne pas pénaliser les variantes haut de gamme légitimes :
- Une BMW Série 3 existe en version de base (~35k $) et en version M3 (~70k $).
- Si on comparait à `MSRP_mean` (~50k $), on supprimerait des annonces M3 à 60k $ qui sont parfaitement normales.
- En comparant à `MSRP_max` (70k $), on ne supprime que les annonces qui dépassent même la version la plus chère neuve.

**Limitations connues**

- Le dataset MSRP couvre les années ~1990 à 2017 → les véhicules `year < 1990` ou `year > 2017` ne sont pas filtrables par cette règle (on les conserve, gérés par les filtres précédents).
- Certaines marques Craigslist n'ont pas de correspondance MSRP (marques rares, fabricants exotiques) → on les conserve aussi.
- Effet de bord assumé : quelques voitures de collection (modèles iconiques avec une côte qui dépasse leur prix neuf d'origine, ex. Mustang 1969 GT500) seront supprimées. Ce parti pris est cohérent avec l'objectif business : prédire le prix d'occasion **standard**, pas la côte du marché de la collection.
Permet aussi l'aprentissage via la RMSE en sortant des valeurs abherrantes par nature tout en pénalisant les gros écarts du modèle quand il se trompent sur les véhicules de masse 

### 2.4 Gestion des valeurs manquantes

| Colonne | Stratégie | Justification |
|---|---|---|
| `odometer` | Imputation par la **médiane globale** | Variable critique, ~1 % de NaN ; la médiane est robuste aux outliers |Idée d'amélioration a terme via du machine learning unsupervised pour créer des valeurs plus cohérentes une deuxième option serait de drop ces lignes car variable critique et facilement forcable a remplir pour l'utilisateur 
| `manufacturer`, `year` | **Drop des lignes** | Variables indispensables pour modéliser et matcher MSRP |
| `model` | Catégorie `"unknown"` | Cardinalité énorme (~10k modalités), pas de mode pertinent à imposer |
| `condition`, `cylinders`, `size`, `drive`, `type`, `fuel`, `transmission`, `paint_color`, `title_status` | Catégorie `"unknown"` | Préserve l'information "non renseigné", évite de fausser les distributions |
| `lat`, `long` | Imputation par la **médiane par `state`** | Préserve la cohérence régionale ; fallback sur la médiane globale si tout NaN dans un state |
| `posting_date` | Imputation par la date médiane | Faible volume de NaN |
| `region`, `state` | Catégorie `"unknown"` | Conservés pour analyse géographique |

---

## 3. Nouvelles features créées

On crée **9 nouvelles features**, regroupées en 4 catégories : temporelles, texte (extraites de `description`), géographique, et marque.

### 3.1 Features temporelles — la mathématique de l'usure

| Feature | Définition | Intuition |
|---|---|---|
| `car_age` | `2021 - year` | Variable plus directement utilisable que `year` ; courbe de dépréciation typique d'un véhicule |
| `miles_per_year` | `odometer / max(car_age, 1)` | Proxy d'**intensité d'usage** : une voiture de 10 ans avec 50 000 miles a dormi au garage, une voiture de 2 ans avec 50 000 miles a fait du VTC. Indicateur d'usure massif. |
| `posting_month` | Mois de l'annonce | Capture la saisonnalité du marché : les cabriolets se vendent plus cher en mai qu'en novembre, les SUVs/4×4 sont recherchés avant l'hiver |
| `posting_year` | Année de l'annonce | Capture une éventuelle dérive temporelle (inflation, effet COVID sur le marché de l'occasion en 2020-2021) |

### 3.2 Features texte — extraites de la colonne `description` par regex

La colonne `description` est un texte libre rempli par le vendeur. On utilise des expressions régulières (insensibles à la casse) avec `pandas.Series.str.contains(...)` pour créer 3 variables :

| Feature | Mots-clés détectés | Intuition |
|---|---|---|
| `is_first_owner` | `1st owner`, `one owner`, `first owner`, `single owner`, `original owner` | Une **première main** se vend toujours plus cher (un seul historique d'usage, traçabilité claire) |
| `has_service_history` | `service records`, `service history`, `dealer maintained`, `maintenance records`, `recent service`, `recently serviced`, `new tires`, `new brakes`, `just serviced`, `all maintenance`, `service receipts` | Un **historique d'entretien clair** rassure l'acheteur et tire le prix vers le haut |
| `desc_length` | Nombre de mots dans `description` | Hypothèse : un vendeur qui prend le temps de rédiger une longue description **détaille un véhicule de meilleure qualité** ou justifie un prix plus élevé |

Une fois ces features extraites, la colonne `description` est supprimée pour libérer la mémoire (texte volumineux non utilisé en aval).

### 3.3 Features géographiques — Rust Belt

| Feature | Définition | Intuition |
|---|---|---|
| `is_rust_belt` | `1` si `state ∈ {MI, OH, IN, IL, PA, NY, WV, WI}` | Les voitures du **Rust Belt** subissent la neige et le sel routier en hiver, ce qui fait **rouiller les châssis** et fait baisser les prix. Effet de décote géographique attendu. |

### 3.4 Feature de marque

| Feature | Définition | Intuition |
|---|---|---|
| `is_premium_brand` | `1` si la marque ∈ `{bmw, mercedes-benz, audi, lexus, porsche, acura, infiniti, jaguar, land rover, tesla, cadillac, lincoln, genesis, alfa-romeo}` | Effet positif fort attendu sur le prix indépendamment du modèle exact |

---

## 4. Transformations appliquées (encoding & scaling)

### 4.1 Stratégie d'encoding

| Type d'encodage | Colonnes | Justification |
|---|---|---|
| **Target encoding** (smoothed mean) | `manufacturer` | ~40 modalités, fort signal sur le prix (BMW vs Hyundai). Le smoothing (`alpha=20`) évite l'overfit sur les marques peu représentées. |
| **Frequency encoding** | `model`, `region` | Cardinalité très élevée (10k+ pour `model`) → one-hot ferait exploser la dimension (mémoire et risque de surapprentissage) |
| **One-hot encoding** (`drop_first=True`) | `condition`, `fuel`, `transmission`, `drive`, `size`, `type`, `paint_color`, `cylinders`, `title_status`, `state` | Cardinalité faible/moyenne (≤ 50) → one-hot reste raisonnable. `drop_first` évite la multicolinéarité parfaite pour les modèles linéaires. |

**Important** : tous les encoders sont **fit sur `X_train` uniquement**, puis appliqués à `X_test`. Le split train/test est fait *avant* tout encodage, pour éviter toute fuite d'information du test dans les statistiques d'encodage (notamment pour le target encoding).

### 4.2 Scaling

`StandardScaler` (centrage/réduction) appliqué aux variables **réellement numériques continues** :
`year`, `age`, `odometer`, `lat`, `long`, `manufacturer_te`, `model_freq`, `region_freq`, `odometer_per_year`, `posting_year`, `posting_month`.

Les colonnes one-hot (déjà dans `[0, 1]`) et `is_premium_brand` (binaire) **ne sont pas scalées**.

Comme pour les encoders, le scaler est `fit` sur train et `transform` sur train+test.

---

## 5. Justification des choix & alternatives testées

### 5.1 Filtre MSRP : pourquoi la règle stricte `price > MSRP_max` ?

| Seuil testé | Décision |
|---|---|
| `price > MSRP_max` (strict) ✅ | **Retenu** : règle métier directement applicable, sans paramètre arbitraire |
| `price > 1.3 × MSRP_max` (tolérant +30 %) | **Rejeté** : laisse passer des annonces clairement gonflées (ex. Honda Civic 2015 à 30k $ alors que le MSRP max est ~22k $) |
| `price > 2.0 × MSRP_max` (laxiste) | **Rejeté** : laisse passer trop d'arnaques évidentes |

La règle stricte (seuil 1.0) correspond directement à l'intuition métier : si une voiture d'occasion coûte plus cher que la même voiture neuve, c'est une aberration. L'utilisation de `MSRP_max` (et non `MSRP_mean`) suffit déjà à protéger les variantes haut de gamme légitimes — il n'est donc pas nécessaire d'ajouter une marge supplémentaire. Le but qui plus est est d'éviter que quelqu'un pense que sa voiture valent plus qu'une voiture neuve ca créerait du doute sur le modèle

### 5.2 Encoding de `model` : alternatives écartées

| Alternative | Pourquoi écartée |
|---|---|
| **One-hot encoding** | ~10 000 colonnes binaires → mémoire et temps d'entraînement explosent, fort risque d'overfit |
| **Target encoding** | Risque de fuite encore plus élevé que sur `manufacturer` (modalités à très faible effectif). Alternative pour itération future avec K-fold target encoding. |
| **Hashing trick** | Plus complexe à interpréter, peu d'avantage ici vs frequency encoding |
| **Fuzzy matching pour normaliser les variantes** (`f-150` ↔ `f150` ↔ `ford f150`) | **Hors scope de cet assignment** mais identifié comme amélioration future (peut être implémenté avec `rapidfuzz` dans une itération ultérieure). |

### 5.3 Imputation des NaN catégoriels par `"unknown"` plutôt que par le mode

- Imputer par le mode introduit un biais : on "vote" artificiellement pour la modalité majoritaire.
- La présence de NaN n'est probablement pas aléatoire (les vendeurs honnêtes remplissent plus de champs) → la "non-information" est elle-même un signal.
- Les modèles d'arbres exploitent naturellement la modalité `"unknown"` comme un split pertinent.

### 5.4 Pas de transformation `log(price)` à ce stade

J'ai choisi de **ne pas appliquer `log(price)` à la target** dans cette pipeline pour deux raisons :

1. Cela rendrait l'évaluation business (MAE en $) moins directe (il faudrait inverser la transformation).
2. Le filtrage des outliers via MSRP a déjà fortement réduit l'asymétrie de la distribution.

Cette transformation pourra être testée dans l'assignment 3 (modélisation) si les modèles linéaires sous-performent.

---

## 6. Impact attendu des transformations sur les modèles

| Transformation | Impact attendu sur modèles linéaires (LinReg, Ridge, Lasso) | Impact attendu sur modèles d'arbres (RF, XGBoost, LightGBM) |
|---|---|---|
| Filtrage outliers MSRP | **Fort** : la régression linéaire est très sensible aux outliers → meilleur fit sur la masse | **Modéré** : les arbres sont robustes mais bénéficient quand même d'une cible plus propre |
| Target encoding `manufacturer` | **Fort** : remplace une variable catégorielle par un signal numérique directement informatif | **Modéré** : les arbres peuvent gérer les catégories one-hot, mais target encoding réduit la profondeur nécessaire |
| Frequency encoding `model` | **Modéré** : capture la popularité (les modèles populaires ont des prix plus stables) | **Modéré** |
| One-hot encoding | Indispensable pour les linéaires | Bénéfique mais alternatives possibles (LightGBM gère les `category` natives) |
| Standard scaling | **Indispensable** pour la régularisation L1/L2 | **Aucun impact** (invariance d'échelle) |
| Feature `car_age` | Permet une relation linéaire plus directe que `year` | Légère amélioration, les arbres peuvent retrouver l'info |
| Feature `miles_per_year` | Capture l'interaction kilométrage/âge | Aide les arbres peu profonds, peu d'impact sur des arbres profonds |
| Feature `is_premium_brand` | Effet additif fort pour les linéaires | Redondant avec target encoding, mais ne nuit pas |
| Features texte (`is_first_owner`, `has_service_history`, `desc_length`) | Variables binaires/numériques directement injectables, signal supplémentaire non capturé par les autres features | Apport modéré ; les arbres peuvent les utiliser dans des splits combinés (ex. âge faible + 1ʳᵉ main → premium) |
| Feature `is_rust_belt` | Effet additif négatif simple (-X $ si rust belt) | Capture une interaction géo difficile à obtenir autrement avec `state` en one-hot |

**Hypothèse globale** : grâce au filtrage MSRP et au target encoding, on s'attend à ce que **les modèles linéaires régularisés (Ridge / ElasticNet) atteignent une MAE compétitive** avec les modèles d'arbres, ce qui n'aurait pas été le cas sur les données brutes.

---

## 7. Exécution & reproductibilité

### 7.1 Comment générer le dataset processé

```bash
# Depuis la racine du projet
jupyter nbconvert --to notebook --execute notebooks/02_cleaning.ipynb
# OU ouvrir le notebook dans VS Code / Jupyter et exécuter toutes les cellules
```

Le notebook produit `data/vehicles_processed.parquet` (~50 MB après encoding).

### 7.2 Comment charger les données dans le code

```python
from src.data import load_dataset_split

X_train, X_test, y_train, y_test = load_dataset_split()
# Toutes les features sont numériques (int / float), prêtes à passer à model.predict(X)
```

### 7.3 Stockage dans le repository

| Type | Emplacement | Versionné ? |
|---|---|---|
| CSV bruts | `data/vehicles.csv`, `data/new_cars_msrp.csv` | ❌ (gitignored — trop volumineux) |
| Parquet processé | `data/vehicles_processed.parquet` | ❌ (gitignored — régénérable depuis le notebook) |
| Notebook de cleaning | `notebooks/02_cleaning.ipynb` | ✅ |
| Module de chargement | `src/data.py` | ✅ |

> Pour un nouveau clone du repo, il faut télécharger les CSV sources depuis Kaggle (les liens sont dans `assignment1.md` et `assignment2.md`) puis exécuter le notebook 02.

---

## 8. Synthèse — chiffres clés

| Étape | Lignes restantes |
|---|---|
| Dataset Craigslist brut | ~426 000 |
| Après filtres basiques (`price`, `year`, `manufacturer`, `odometer`) | ~360 000 |
| Après filtrage outliers MSRP (règle stricte `price > MSRP_max`) | à confirmer après exécution |
| **Dataset final** (après imputation NaN) | **à confirmer après exécution** |

Train / test split : **80 / 20** (`random_state=42`), soit ~284k / 71k lignes.

Nombre de features finales (après encoding) : **~80 colonnes** (variable selon les modalités one-hot effectivement présentes dans le train).
