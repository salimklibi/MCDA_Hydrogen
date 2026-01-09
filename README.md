# Analyse MCDA Multi-Scénarios Hydrogène Vendée avec badges Python 3.10+, PyMCDM, pandas, matplotlib/seaborn via shields.io .

# Description
Ce script principal effectue une analyse multi-critères décisionnelle (MCDA) pour comparer trois scénarios H2 en Vendée : production 100% locale (type A), import régional (type B), import national/UE (type C). Il applique SAW (toujours), TOPSIS et PROMETHEE II (si PyMCDM installé) sur critères coût, GES, emploi, acceptabilité, avec poids par acteurs (industriels, collectivités, etc.) et horizons de sensibilité (H1, H20, H50, H100).
​

# Installation
Clonez : git clone https://github.com/salimklibi/MCDA_Hydrogen.git

Environnement : conda create -n mcda-h2 python=3.10 && conda activate mcda-h2

Dépendances : pip install pandas numpy matplotlib seaborn pymcdm (optionnel pour TOPSIS/PROMETHEE).
​

# Utilisation
Exécutez python main.py pour lancer automatiquement tous les calculs. Le script crée un dossier output/mcda/ avec :

CSVs par méthode (résultats rangs/scores par horizon/acteurs/scénarios)

PNG individuels (heatmaps, boxplots, évolutions, radars)

COMPARAISONFINALE.png (barres rangs moyens + scatter corrélations méthodes).
​

# Personnalisation
Modifiez alternatives, decisionmatrixH1, types, actors en haut du script pour adapter critères/poids. PyMCDM optionnel : SAW seul sinon.
​

# Outputs Exemples
Rang moyen : scénario A souvent favorisé en H100 pour acceptabilité locale.

Visualisations : heatmaps acteurs vs horizons, corrélations méthodes.
​

# Licence et Contribution
Licence MIT. Lié à votre PhD sur hydrogène à Paris-Dauphine ; contributions via PR bienvenues.
​
