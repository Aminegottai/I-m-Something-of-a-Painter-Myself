# ===============================
# Prédiction des ventes avec Gradient Boosting et RMSLE
# ===============================

# 1️⃣ Import des librairies
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sans affichage pour Windows
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np

# 2️⃣ Charger le dataset
df = pd.read_csv("C:\\Users\\amine\\Downloads\\sales.csv", encoding='latin1')

# 3️⃣ Nettoyage
df['Postal_Code'] = df['Postal_Code'].fillna(df['Postal_Code'].median())

# 4️⃣ Conversion des dates
df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True)
df['Ship_Date'] = pd.to_datetime(df['Ship_Date'], dayfirst=True)

# 5️⃣ Feature engineering temporel
df['Month'] = df['Order_Date'].dt.month
df['Day'] = df['Order_Date'].dt.day
df['Weekday'] = df['Order_Date'].dt.weekday

# 6️⃣ Encodage des colonnes catégorielles
categorical_cols = ['Segment', 'Country', 'City', 'State', 'Region',
                    'Category', 'Sub_Category', 'Ship_Mode']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 7️⃣ Sélection des features et target
X = df.drop(['Sales', 'Order_ID', 'Customer_ID', 'Customer_Name', 
             'Product_ID', 'Product_Name', 'Order_Date', 'Ship_Date', 'Row_ID'], axis=1)
y = df['Sales']

# 8️⃣ Transformation log pour réduire l’impact des valeurs extrêmes
y_log = np.log1p(y)  # log(y + 1)

# 9️⃣ Division train/test
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 🔟 Entraînement du modèle Gradient Boosting
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train_log)

# 1️⃣1️⃣ Prédiction
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Retour à l’échelle originale

# 1️⃣2️⃣ Évaluation avec RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_test_log, y_pred_log))
print("Root Mean Squared Logarithmic Error (RMSLE) :", rmsle)

# 1️⃣3️⃣ Visualisation et sauvegarde du graphique
plt.figure(figsize=(12,6))
plt.plot(np.expm1(y_test_log).values, label='Ventes réelles', marker='o', linestyle='')
plt.plot(y_pred, label='Prédictions', marker='x', linestyle='')
plt.title("Ventes réelles vs prédites (Gradient Boosting)")
plt.xlabel("Échantillons")
plt.ylabel("Ventes")
plt.legend()
plt.savefig("ventes_prediction_gb.png")
print("Graphique sauvegardé sous ventes_prediction_gb.png")
