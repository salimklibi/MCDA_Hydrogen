import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppression des warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Détection PyMCDM
try:
    from pymcdm.methods import TOPSIS, PROMETHEE_II
    from pymcdm.helpers import rrankdata

    PYMCDM_AVAILABLE = True
    print("✅ PyMCDM installé")
except ImportError:
    PYMCDM_AVAILABLE = False
    print("⚠️ PyMCDM absent → SAW uniquement")

# =========================
# CONFIGURATION
# =========================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.dpi': 150, 'savefig.dpi': 300,  # DPI légèrement réduit pour affichage écran, élevé pour save
    'axes.labelsize': 12
})

methods_list = ["SAW"]
if PYMCDM_AVAILABLE:
    methods_list.extend(["TOPSIS", "PROMETHEE"])

# =========================
# DONNÉES
# =========================
alternatives = [
    "Scénario 1 : H2 100% Vendée (type A)",
    "Scénario 2 : H2 Vendée + import régional (type B)",
    "Scénario 3 : Import H2 national/UE (type C)",
]

decision_matrix_H1 = np.array([
    [6.0, 1.0, 120.0, 4.5],
    [5.0, 1.5, 80.0, 4.0],
    [4.0, 3.0, 30.0, 3.0],
], dtype=float)

types = np.array([-1, -1, 1, 1], dtype=int)

actors = {
    "Industriels_Finance": np.array([0.35, 0.30, 0.20, 0.15]),
    "Collectivites": np.array([0.20, 0.30, 0.25, 0.25]),
    "ONG_Locales": np.array([0.10, 0.50, 0.15, 0.25]),
    "Autorites": np.array([0.25, 0.30, 0.20, 0.25]),
    "Scientifiques": np.array([0.25, 0.25, 0.25, 0.25]),
}
for k, w in actors.items():
    actors[k] = w / w.sum()


def build_horizons(M_H1):
    # (Fonction inchangée)
    M_H1 = M_H1.copy()
    M_H20 = M_H1.copy()
    M_H20[0, 0] *= 0.85;
    M_H20[1, 0] *= 0.90;
    M_H20[2, 0] *= 0.95
    M_H20[0, 1] *= 0.90;
    M_H20[1, 1] *= 0.90;
    M_H20[2, 1] *= 0.60
    M_H20[0, 2] *= 1.05;
    M_H20[1, 2] *= 1.20;
    M_H20[2, 2] *= 1.10
    M_H20[0, 3] *= 1.05;
    M_H20[1, 3] *= 1.15;
    M_H20[2, 3] *= 1.02

    M_H50 = M_H20.copy()
    M_H50[0, 0] *= 0.90;
    M_H50[1, 0] *= 0.90;
    M_H50[2, 0] *= 0.80
    M_H50[0, 1] *= 0.80;
    M_H50[1, 1] *= 0.75;
    M_H50[2, 1] *= 0.70
    M_H50[0, 2] *= 1.05;
    M_H50[1, 2] *= 1.15;
    M_H50[2, 2] *= 1.20
    M_H50[0, 3] *= 1.05;
    M_H50[1, 3] *= 1.10;
    M_H50[2, 3] *= 1.10

    M_H100 = M_H50.copy()
    M_H100[0, 0] *= 0.90;
    M_H100[1, 0] *= 0.90;
    M_H100[2, 0] *= 0.90
    M_H100[0, 1] *= 0.70;
    M_H100[1, 1] *= 0.70;
    M_H100[2, 1] *= 0.70
    M_H100[0, 2] *= 1.05;
    M_H100[1, 2] *= 1.05;
    M_H100[2, 2] *= 1.10
    M_H100[0, 3] *= 1.05;
    M_H100[1, 3] *= 1.05;
    M_H100[2, 3] *= 1.15
    return {"H1": M_H1, "H20": M_H20, "H50": M_H50, "H100": M_H100}


# Moteurs MCDA
def run_saw(M, w, t):
    M_norm = M.copy().astype(float)
    for j in range(M_norm.shape[1]):
        col = M_norm[:, j]
        if abs(col.max() - col.min()) < 1e-10: M_norm[:, j] = 0.0; continue
        if t[j] == 1:
            M_norm[:, j] = (col - col.min()) / (col.max() - col.min())
        else:
            M_norm[:, j] = (col.max() - col) / (col.max() - col.min())
    prefs = M_norm @ w
    ranks = np.argsort(-prefs).astype(int) + 1
    return prefs, ranks


def run_topsis(M, w, t):
    model = TOPSIS()
    prefs = model(M, w, t)
    ranks = rrankdata(prefs)
    return prefs, ranks


def run_promethee(M, w, t):
    model = PROMETHEE_II("usual")
    prefs = model(M, w, t)
    ranks = (-prefs).argsort().astype(int) + 1
    return prefs, ranks


def run_method(M, w, t, method):
    m = method.upper()
    if m == "SAW":
        return run_saw(M, w, t)
    elif m == "TOPSIS":
        return run_topsis(M, w, t)
    elif m == "PROMETHEE":
        return run_promethee(M, w, t)
    else:
        raise ValueError("Méthode inconnue")


# =========================
# FONCTIONS DE PLOT
# =========================
def plot_individual_method(df, method_name):
    """Génère et sauvegarde les plots pour une méthode donnée"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Résultats détaillés : {method_name}', fontsize=16, fontweight='bold')

    # 1. Heatmap Rangs
    pivot_avg = df.pivot_table('Rang', 'Acteur', 'Horizon', 'mean')
    sns.heatmap(pivot_avg, annot=True, cmap='RdYlGn_r', fmt='.1f', center=2, linewidths=.5, ax=axes[0, 0])
    axes[0, 0].set_title('Rang Moyen (Acteurs vs Horizons)')

    # 2. Boxplot Rangs par Scénario
    sns.boxplot(data=df, x='Horizon', y='Rang', hue='Scénario', ax=axes[0, 1])
    axes[0, 1].set_title('Distribution des Rangs')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. Evolution Score Moyen
    for scenario in df['Scénario'].unique():
        subset = df[df['Scénario'] == scenario]
        avg_scores = subset.groupby('Horizon')['Score'].mean()
        axes[1, 0].plot(avg_scores.index, avg_scores.values, marker='o', label=scenario.split(':')[1].strip())
    axes[1, 0].set_title('Évolution Score Moyen')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Radar H100 (Statique)
    # Note: On refait la normalisation pour afficher les perfs brutes, pas les scores MCDA
    M_h100 = horizons['H100']
    M_norm = M_h100 / M_h100.max(axis=0)
    categories = ['Coût', 'GES', 'Emploi', 'Acceptabilité']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    ax_radar = plt.subplot(2, 2, 4, polar=True)
    for i, alt in enumerate(alternatives):
        values = M_norm[i].tolist() + [M_norm[i, 0]]
        ax_radar.plot(angles, values, linewidth=2, label=alt.split(':')[1].strip())
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax_radar.set_title('Performance H100 (Indépendant de la méthode)')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(f'output_mcda/individuel_{method_name}.png', bbox_inches='tight')
    plt.close()  # Important pour éviter de garder 50 fenêtres ouvertes


# =========================
# PROGRAMME PRINCIPAL
# =========================
horizons = build_horizons(decision_matrix_H1)
os.makedirs('output_mcda', exist_ok=True)

all_results_dfs = []  # Pour stocker les dataframes de chaque méthode

print("\n" + "=" * 60)
print("🚀 LANCEMENT AUTOMATIQUE DES CALCULS MULTI-MÉTHODES")
print("=" * 60)

for method in methods_list:
    print(f"\n⏳ Traitement de {method}...")

    method_results = []
    for h_name, M in horizons.items():
        for actor_name, weights in actors.items():
            prefs, ranks = run_method(M, weights, types, method)
            for i, (alt, pref, rank) in enumerate(zip(alternatives, prefs, ranks)):
                method_results.append({
                    'Horizon': h_name, 'Acteur': actor_name, 'Scénario': alt,
                    'Score': float(pref), 'Rang': int(rank), 'Méthode': method
                })

    df_method = pd.DataFrame(method_results)

    # 1. Enregistrer CSV
    csv_path = f'output_mcda/results_{method.lower()}.csv'
    df_method.to_csv(csv_path, index=False)
    print(f"   💾 CSV sauvegardé : {csv_path}")

    # 2. Faire les plots individuels
    plot_individual_method(df_method, method)
    print(f"   📊 Plots sauvegardés : output_mcda/individuel_{method}.png")

    # 3. Ajouter au pool global pour la comparaison
    all_results_dfs.append(df_method)

# Concaténation de tous les résultats
df_all = pd.concat(all_results_dfs)

# =========================
# 4. PLOT DE COMPARAISON FINALE
# =========================
print("\n⏳ Génération du graphique de comparaison...")

fig_comp, axes_comp = plt.subplots(1, 2, figsize=(16, 6))
fig_comp.suptitle('Comparaison Synthétique des Méthodes MCDA', fontsize=16, fontweight='bold')

# A. Barplot du Rang Moyen Global (Tous acteurs/horizons confondus)
# Cela montre quelle méthode favorise quel scénario en moyenne
avg_ranks = df_all.groupby(['Méthode', 'Scénario'])['Rang'].mean().reset_index()
# On reformate pour seaborn
avg_ranks['Scénario_Short'] = avg_ranks['Scénario'].apply(lambda x: x.split(':')[1].strip())

sns.barplot(data=avg_ranks, x='Méthode', y='Rang', hue='Scénario_Short',
            palette='Set2', ax=axes_comp[0])
axes_comp[0].set_title('Rang Moyen (plus bas = meilleur)', fontweight='bold')
axes_comp[0].set_ylim(0, 3.5)
axes_comp[0].legend(title='Scénario')

# B. Scatter Plot : Corrélation des scores (ex: SAW vs TOPSIS)
if len(methods_list) >= 2:
    m1 = methods_list[0]
    m2 = methods_list[1]

    # On moyenne les scores par horizon et scénario pour simplifier le scatter
    pivot_s1 = df_all[df_all['Méthode'] == m1].groupby(['Horizon', 'Scénario'])['Score'].mean().reset_index()
    pivot_s2 = df_all[df_all['Méthode'] == m2].groupby(['Horizon', 'Scénario'])['Score'].mean().reset_index()

    # Merge sur les colonnes communes
    merged = pd.merge(pivot_s1, pivot_s2, on=['Horizon', 'Scénario'], suffixes=('_' + m1, '_' + m2))

    sns.scatterplot(data=merged, x=f'Score_{m1}', y=f'Score_{m2}',
                    hue='Scénario', style='Horizon', s=100, ax=axes_comp[1])

    # Ligne diagonale
    min_val = min(merged[f'Score_{m1}'].min(), merged[f'Score_{m2}'].min())
    max_val = max(merged[f'Score_{m1}'].max(), merged[f'Score_{m2}'].max())
    axes_comp[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

    axes_comp[1].set_title(f'Corrélation Scores : {m1} vs {m2}')
    axes_comp[1].set_xlabel(f'Score Normalisé ({m1})')
    axes_comp[1].set_ylabel(f'Score Normalisé ({m2})')
    axes_comp[1].grid(True, alpha=0.3)
else:
    axes_comp[1].text(0.5, 0.5, "Pas assez de méthodes pour comparer", ha='center')

plt.tight_layout()
plt.savefig('output_mcda/COMPARAISON_FINALE.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ TERMINÉ !")
print("📂 Dossiers générés dans 'output_mcda/' :")
print("  - CSVs pour chaque méthode")
print("  - Images individuelles pour chaque méthode")
print("  - 'COMPARAISON_FINALE.png' (Synthèse)")