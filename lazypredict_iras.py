"""
Vigilancia epidemiológica de IRA (Perú, 2000-2025) - Datos Abiertos
PROBLEMA: predecir si un distrito tendrá al menos 1 caso de neumonía en
menores de 5 años en una semana epidemiológica (alerta temprana), usando
solo la información de IRA no neumónica, la ubicación y el calendario.
Comparación de modelos con LazyPredict y selección del mejor.

Instalación:  pip install lazypredict scikit-learn pandas numpy matplotlib seaborn
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from lazypredict.Supervised import LazyClassifier, CLASSIFIERS

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
from pathlib import Path
# Carpeta Descargas del usuario actual (Windows: C:\\Users\\<usuario>\\Downloads)
RUTA_CSV = Path.home() / "Downloads" / "datos_abiertos_vigilancia_iras_2000_2025_2.csv"
N_TRAIN, N_TEST = 60_000, 20_000     # tamaño de muestra (LazyPredict es lento con millones de filas)

# ----------------------------------------------------------------------
# 1. Carga
# ----------------------------------------------------------------------
df = pd.read_csv(RUTA_CSV, sep=";", encoding="utf-8-sig")
print("Dimensiones:", df.shape)
print(df.isna().sum().sum(), "valores nulos")

# ----------------------------------------------------------------------
# 2. EDA rápido
# ----------------------------------------------------------------------
df["tiene_neumonia"] = (df["neumonias_men5"] > 0).astype(int)   # variable objetivo
print("\nProporción de semanas-distrito con neumonía en <5 años:")
print(df["tiene_neumonia"].value_counts(normalize=True).round(4))

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
df.groupby("ano")["tiene_neumonia"].mean().plot(ax=ax[0], marker="o")
ax[0].set_title("Proporción con neumonía (<5 años) por año")
df.groupby("semana")["tiene_neumonia"].mean().plot(ax=ax[1], marker="o")
ax[1].set_title("Estacionalidad: proporción por semana epidemiológica")
plt.tight_layout(); plt.savefig("eda_iras.png", dpi=120); plt.show()

# ----------------------------------------------------------------------
# 3. Ingeniería de características (sin usar otras variables de resultado
#    como neumonías/hospitalizados/defunciones, para evitar fuga de datos)
# ----------------------------------------------------------------------
df = df.sort_values(["ubigeo", "ano", "semana"]).reset_index(drop=True)
g = df.groupby("ubigeo")["ira_no_neumonia"]
df["ira_lag1"] = g.shift(1)
df["ira_lag2"] = g.shift(2)
df["ira_media4"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
df["semana_sin"] = np.sin(2 * np.pi * df["semana"] / 53)
df["semana_cos"] = np.cos(2 * np.pi * df["semana"] / 53)
df[["ira_lag1", "ira_lag2", "ira_media4"]] = df[["ira_lag1", "ira_lag2", "ira_media4"]].fillna(0)

# Departamento -> variables dummy
df = pd.concat([df, pd.get_dummies(df["departamento"], prefix="dep", dtype=int)], axis=1)

features = (["ira_no_neumonia", "ira_lag1", "ira_lag2", "ira_media4",
             "ano", "semana", "semana_sin", "semana_cos", "sub_reg_nt"]
            + [c for c in df.columns if c.startswith("dep_")])

# ----------------------------------------------------------------------
# 4. Partición TEMPORAL (entrena con el pasado, prueba con el futuro)
# ----------------------------------------------------------------------
train = df[df["ano"] <= 2022]
test = df[df["ano"] >= 2023]
train = train.sample(min(N_TRAIN, len(train)), random_state=RANDOM_STATE)
test = test.sample(min(N_TEST, len(test)), random_state=RANDOM_STATE)

X_train, y_train = train[features], train["tiene_neumonia"]
X_test, y_test = test[features], test["tiene_neumonia"]
print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# ----------------------------------------------------------------------
# 5. LazyPredict (se omiten modelos muy lentos con datos de este tamaño)
# ----------------------------------------------------------------------
LENTOS = {"SVC", "NuSVC", "LabelSpreading", "LabelPropagation",
          "KNeighborsClassifier", "GaussianProcessClassifier"}
clasificadores = [n for n, c in CLASSIFIERS if n not in LENTOS]

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True,
                     classifiers=clasificadores, random_state=RANDOM_STATE)
modelos, predicciones = clf.fit(X_train, X_test, y_train, y_test)

# Con clases desbalanceadas (~15% positivos) NO conviene elegir por Accuracy ni
# por el F1 ponderado (lo domina la clase mayoritaria). Se ordena por ROC AUC
# y se revisa también Balanced Accuracy.
modelos = modelos.sort_values("ROC AUC", ascending=False)
print("\n=== Ranking de modelos ===")
print(modelos.head(15))
modelos.to_csv("ranking_lazypredict_iras.csv")

# ----------------------------------------------------------------------
# 6. Mejor modelo
# ----------------------------------------------------------------------
mejor = modelos.index[0]
print(f"\n>>> Mejor modelo según ROC AUC: {mejor}")
print(modelos.iloc[0])

y_pred = predicciones[mejor]
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=["sin neumonía", "con neumonía"]))

# Gráfico comparativo
top = modelos.head(10).reset_index()
top = top.rename(columns={"Model": "Modelo"})
plt.figure(figsize=(9, 5))
sns.barplot(data=top, y="Modelo", x="ROC AUC", color="steelblue")
plt.title("Top 10 modelos (LazyPredict) - ROC AUC")
plt.tight_layout(); plt.savefig("top_modelos_iras.png", dpi=120); plt.show()