# Analyse de sentiment : avis Yelp

Ce projet de traitement du langage naturel classe un avis Yelp rédigé en anglais comme **positif** ou **négatif**. Le [notebook](projet_ml_yelp.ipynb) compare trois classifieurs de texte et un réseau récurrent LSTM. L’application Streamlit permet d’utiliser le modèle retenu sur un nouvel avis. Le SVM fourni avec le projet a correctement classé **92,32 % des 38 000 avis de test** lors de l’évaluation.

## Lancer l’application avec le modèle fourni

Ouvrez un terminal dans le dossier contenant `app.py`. L’application peut démarrer avec le modèle enregistré, sans charger les CSV ni relancer l’entraînement. Pour créer un environnement Python 3.11 avec Conda, puis lancer Streamlit :

```bash
conda create -n yelp-avis python=3.11 pip -y
conda activate yelp-avis
python -m pip install "streamlit>=1.50,<2" "scikit-learn==1.9.1" joblib
python -m streamlit run app.py
```

Si un environnement Python 3.11 est déjà actif, commencez à la commande d’installation avec `pip`. La version de scikit-learn indiquée correspond à celle du modèle sauvegardé. Ouvrez l’adresse locale affichée dans le terminal. `Ctrl+C` arrête le serveur.

### Analyser un avis

1. Saisissez un avis en anglais dans **Votre texte**, ou importez un fichier `.txt` en UTF-8. Si les deux champs sont remplis, l’application analyse le fichier.
2. Cliquez sur **Analyser**. La catégorie affichée est **Positif** dès que la probabilité positive atteint 50 %. En dessous de ce seuil, elle est **Négatif**.
3. Lisez le pourcentage et la barre sous la catégorie. Ils correspondent à la **classe annoncée**. Le détail en dessous donne les probabilités des deux classes.

Deux exemples vérifiés avec le modèle fourni :

| Texte saisi | Catégorie affichée | Probabilité associée |
| --- | --- | ---: |
| `it is good.` | Positif | 99,9 % |
| `I have zero complaints about the service.` | Négatif | 82,5 % |

Le second exemple illustre une erreur. La personne écrit qu’elle n’a aucun reproche à faire au service. Or le SVM représente ici chaque mot séparément et peut perdre le sens de l’expression « zero complaints ». Même une prédiction accompagnée d’un pourcentage élevé peut être incorrecte.

## Reproduire le TP dans le notebook

**Dans VS Code**, ouvrez le notebook depuis la racine du projet avec l’extension Jupyter et sélectionnez votre environnement Python 3.11 comme noyau. Si l’environnement vient d’être créé, installez-y le noyau avec `python -m pip install ipykernel`. La première cellule du notebook installe les bibliothèques d’analyse et d’entraînement, dont TensorFlow.

**Dans Google Colab**, importez le même notebook. La cellule de chargement cherche les CSV, puis l’archive locale. Si ces sources manquent, elle télécharge l’archive d’environ 166 Mo. Un GPU est facultatif. [Archive Yelp Review Polarity utilisée par le notebook](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz)

Choisissez la valeur de `MODE_RAPIDE` dans la cellule de chargement avant d’exécuter la suite dans l’ordre.

| Paramètre | Mode rapide (`True`) | Mode complet (`False`, par défaut) |
| --- | ---: | ---: |
| Avis d’entraînement chargés pour l’exploration | 2 500 | 560 000 |
| Avis disponibles pour ajuster les modèles | 2 500 | 100 000 |
| Avis de test | 600 | 38 000 |
| Époques du LSTM | 1 | 3 |

Le mode rapide permet de vérifier toute la chaîne. Le mode complet sert à comparer les modèles. La partie 4 sauvegarde le gagnant de l’exécution en cours, y compris en mode rapide. La partie 5 régénère `app.py` et démarre Streamlit. Pour essayer l’interface, laissez le noyau actif et exécutez la cellule `serveur_yelp.terminate()` seulement après votre essai. Dans Colab, la partie 5 propose aussi une archive ZIP contenant l’application et son modèle.

## Données et choix de modélisation

Les fichiers CSV contiennent chacun deux colonnes sans en-tête. La première donne le sentiment (`1` pour négatif, `2` pour positif), la seconde contient le texte. Le notebook recode les classes en `0` et `1` pour l’apprentissage. Dans le jeu d’entraînement complet, les deux sentiments sont également représentés. Prédire toujours la même classe y donnerait donc 50 % de bonnes réponses.

| Étape | Choix et justification |
| --- | --- |
| Nettoyage | Passage en minuscules, remplacement de la ponctuation et des chiffres par des espaces, puis normalisation des espaces. L’application applique les mêmes transformations aux nouveaux avis. |
| Représentation du texte | Avec Naive Bayes, le comptage brut des mots donne 86,72 % de bonnes réponses, contre 87,65 % avec TF-IDF. Le TP retient TF-IDF, limité à 20 000 termes, pour comparer les modèles classiques. |
| Modèles classiques | Le notebook fait varier le lissage de Naive Bayes, la régularisation du SVM et le nombre d’arbres de la forêt aléatoire. Le vocabulaire et les poids TF-IDF sont appris sur l’entraînement. |
| LSTM | La médiane est de 97 mots, et 80,3 % des avis bruts comptent au plus 200 mots. Le réseau conserve les 200 premières positions après tokenisation. Il utilise des vecteurs de 64 dimensions, un LSTM de 64 unités et 10 % des séquences d’entraînement pour la validation. |

## Résultats de l’exécution complète

Pour chaque famille classique, le notebook garde le réglage offrant le meilleur **F1 de la classe positive**. Ce score combine la précision des prédictions positives et la proportion d’avis positifs retrouvés. Il compare ensuite les quatre finalistes sur les mêmes avis de test et choisit le gagnant selon l’**accuracy**, la proportion totale d’avis correctement classés.

| Modèle | Réglage retenu | Accuracy | F1 positif |
| --- | --- | ---: | ---: |
| **SVM linéaire + TF-IDF** | **`C=0.1`** | **92,32 %** | **0,9232** |
| LSTM | 200 positions, 3 époques | 90,88 % | 0,9091 |
| Random Forest + TF-IDF | 100 arbres | 88,79 % | 0,8858 |
| Naive Bayes + TF-IDF | `alpha=1.0` | 87,65 % | 0,8768 |

Le SVM fait **2 917 erreurs** sur le test. Il classe 1 438 avis négatifs comme positifs et 1 479 avis positifs comme négatifs. Ces scores proviennent du tableau de comparaison, avant la calibration pour l’application. Le test ayant aussi servi à départager les réglages, une évaluation indépendante du modèle retenu demanderait un autre jeu d’avis.

### D’où viennent les probabilités du SVM ?

Le SVM produit d’abord un score de décision. Pour afficher un pourcentage, le notebook ajuste une sigmoïde sur **5 000 autres avis du fichier d’entraînement**, exclus de l’ajustement de TF-IDF et du SVM. En mode rapide, il utilise une validation croisée à trois plis pour obtenir les scores de calibration. Le pourcentage affiché concerne l’avis saisi. Les 92,32 % du tableau mesurent, eux, la réussite globale sur le jeu de test.

## Fichiers du projet

| Fichier | Rôle |
| --- | --- |
| [projet_ml_yelp.ipynb](projet_ml_yelp.ipynb) | Explore les données, entraîne et compare les modèles, explique les graphiques, puis génère l’application. |
| [app.py](app.py) | Charge le modèle sauvegardé et affiche la prédiction dans Streamlit. |
| `model_info.pkl` | Indique à l’application la famille et le nom du modèle à charger. |
| `model_sklearn.joblib` | Contient le modèle classique exporté. La version fournie réunit TF-IDF, le SVM et sa calibration. |
| `yelp_review_polarity_csv/` | Contient `train.csv` et `test.csv`. |
| `yelp_review_polarity_csv.tgz` | Archive que le notebook peut lire directement si les CSV sont indisponibles. |

Pour utiliser le SVM ailleurs, copiez `app.py`, `model_info.pkl` et `model_sklearn.joblib` dans le même dossier. Si le LSTM gagne lors d’une prochaine exécution, la partie 4 exporte `rnn_yelp.keras` et `tokenizer.pkl`. L’application les charge alors avec TensorFlow.
