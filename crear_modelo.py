import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Datos de ejemplo de partidos
data = pd.DataFrame({
    'local_goles': [2, 1, 0, 3, 2],
    'visitante_goles': [1, 1, 2, 0, 1],
    'posesion_local': [55, 48, 60, 70, 50],
    'posesion_visitante': [45, 52, 40, 30, 50],
    'result': ['local', 'empate', 'visitante', 'local', 'empate']
})

# Variables independientes y dependiente
X = data.drop(columns=['result'])
y = data['result']

# Entrenamiento del modelo
model = RandomForestClassifier()
model.fit(X, y)

# Guardar el modelo entrenado
joblib.dump(model, 'model.pkl')