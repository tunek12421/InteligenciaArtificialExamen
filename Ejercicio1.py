"""
Ejercicio 1: Clasificación binaria de tumores con modelos clásicos
Dataset: Breast Cancer Wisconsin (Sklearn)
Objetivo: Predecir si un tumor es maligno o benigno
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("EJERCICIO 1: CLASIFICACIÓN BINARIA DE TUMORES")
print("="*60)

# 1. CARGAR Y EXPLORAR DATOS
print("\n1. CARGANDO Y EXPLORANDO DATOS...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(f"Forma del dataset: {X.shape}")
print(f"Clases: {data.target_names}")
print(f"Distribución de clases:")
print(pd.Series(y).value_counts().sort_index())
print(f"  - Maligno (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"  - Benigno (1): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

# Estadísticas descriptivas
print(f"\nEstadísticas descriptivas:")
print(X.describe().iloc[:, :5])  # Primeras 5 características

# 2. PREPROCESAMIENTO
print("\n2. PREPROCESAMIENTO DE DATOS...")
# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalización/Estandarización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Conjunto de entrenamiento: {X_train_scaled.shape}")
print(f"Conjunto de prueba: {X_test_scaled.shape}")

# 3. ENTRENAMIENTO DE MODELOS
print("\n3. ENTRENAMIENTO DE MODELOS...")

# 3.1 Regresión Logística Regularizada
print("\n3.1 Regresión Logística Regularizada...")

# L1 (Lasso)
lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
lr_l1.fit(X_train_scaled, y_train)
y_pred_l1 = lr_l1.predict(X_test_scaled)

# L2 (Ridge)  
lr_l2 = LogisticRegression(penalty='l2', random_state=42)
lr_l2.fit(X_train_scaled, y_train)
y_pred_l2 = lr_l2.predict(X_test_scaled)

print(f"Regresión Logística L1 - Accuracy: {accuracy_score(y_test, y_pred_l1):.4f}")
print(f"Regresión Logística L2 - Accuracy: {accuracy_score(y_test, y_pred_l2):.4f}")

# 3.2 Random Forest
print("\n3.2 Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print(f"Random Forest - Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# 3.3 Naive Bayes
print("\n3.3 Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
print(f"Naive Bayes - Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")

# 3.4 XGBoost (Opcional)
print("\n3.4 XGBoost...")
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
print(f"XGBoost - Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")

# 4. EVALUACIÓN Y COMPARACIÓN
print("\n4. EVALUACIÓN Y COMPARACIÓN DE MODELOS...")

models = {
    'Logistic Regression L1': (lr_l1, y_pred_l1),
    'Logistic Regression L2': (lr_l2, y_pred_l2),
    'Random Forest': (rf, y_pred_rf),
    'Naive Bayes': (nb, y_pred_nb),
    'XGBoost': (xgb_model, y_pred_xgb)
}

results = []
for name, (model, y_pred) in models.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

results_df = pd.DataFrame(results)
print("\nRESULTADOS COMPARATIVOS:")
print(results_df.round(4))

# 5. VALIDACIÓN CRUZADA
print("\n5. VALIDACIÓN CRUZADA...")
for name, (model, _) in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 6. INVESTIGACIÓN DE REGULARIZACIÓN
print("\n6. EFECTO DE LA REGULARIZACIÓN...")

# Probar diferentes valores de C
C_values = [0.01, 0.1, 1, 10, 100]
l1_scores = []
l2_scores = []

for C in C_values:
    # L1
    lr_l1_c = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42)
    cv_l1 = cross_val_score(lr_l1_c, X_train_scaled, y_train, cv=5, scoring='accuracy')
    l1_scores.append(cv_l1.mean())
    
    # L2
    lr_l2_c = LogisticRegression(penalty='l2', C=C, random_state=42)
    cv_l2 = cross_val_score(lr_l2_c, X_train_scaled, y_train, cv=5, scoring='accuracy')
    l2_scores.append(cv_l2.mean())

print("\nEfecto del parámetro C en la regularización:")
reg_results = pd.DataFrame({
    'C': C_values,
    'L1 (Lasso)': l1_scores,
    'L2 (Ridge)': l2_scores
})
print(reg_results.round(4))

# 7. MATRIZ DE CONFUSIÓN DEL MEJOR MODELO
print("\n7. ANÁLISIS DETALLADO DEL MEJOR MODELO...")
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_model, best_pred = models[best_model_name]

print(f"\nMejor modelo: {best_model_name}")
print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, best_pred)
print(cm)

print(f"\nReporte de Clasificación:")
print(classification_report(y_test, best_pred, target_names=data.target_names))

# 8. VISUALIZACIÓN
plt.figure(figsize=(15, 10))

# Subplot 1: Comparación de modelos
plt.subplot(2, 3, 1)
plt.bar(results_df['Model'], results_df['Accuracy'])
plt.title('Comparación de Accuracy por Modelo')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')

# Subplot 2: Efecto de regularización
plt.subplot(2, 3, 2)
plt.plot(C_values, l1_scores, 'o-', label='L1 (Lasso)')
plt.plot(C_values, l2_scores, 's-', label='L2 (Ridge)')
plt.xscale('log')
plt.xlabel('C (Inverso de regularización)')
plt.ylabel('CV Accuracy')
plt.title('Efecto de la Regularización')
plt.legend()

# Subplot 3: Matriz de confusión
plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title(f'Matriz de Confusión - {best_model_name}')
plt.ylabel('Real')
plt.xlabel('Predicho')

# Subplot 4: Métricas por modelo
plt.subplot(2, 3, 4)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df))
width = 0.2
for i, metric in enumerate(metrics):
    plt.bar(x + i*width, results_df[metric], width, label=metric)
plt.xlabel('Modelos')
plt.ylabel('Score')
plt.title('Métricas por Modelo')
plt.xticks(x + width*1.5, [m.replace(' ', '\n') for m in results_df['Model']])
plt.legend()

# Subplot 5: Importancia de características (Random Forest)
plt.subplot(2, 3, 5)
feature_importance = rf.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:]
plt.barh(range(10), feature_importance[top_features_idx])
plt.yticks(range(10), [data.feature_names[i][:15] + '...' if len(data.feature_names[i]) > 15 
                      else data.feature_names[i] for i in top_features_idx])
plt.xlabel('Importancia')
plt.title('Top 10 Características - Random Forest')

plt.tight_layout()
plt.savefig('Ejercicio1_resultados.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("EJERCICIO 1 COMPLETADO")
print("="*60)
print(f"Mejor modelo: {best_model_name}")
print(f"Accuracy: {results_df.loc[results_df['F1-Score'].idxmax(), 'Accuracy']:.4f}")
print(f"F1-Score: {results_df.loc[results_df['F1-Score'].idxmax(), 'F1-Score']:.4f}")
print("Gráfico guardado como: Ejercicio1_resultados.png")