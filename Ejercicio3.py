"""
Ejercicio 3: Clustering con K-Means y reducción de dimensionalidad (PCA)
Dataset: Iris
Objetivo: Agrupar datos no etiquetados y visualizar con PCA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("EJERCICIO 3: CLUSTERING K-MEANS Y REDUCCIÓN PCA")
print("="*60)

# 1. CARGAR Y EXPLORAR DATOS
print("\n1. CARGANDO Y EXPLORANDO DATOS...")
iris = load_iris()
X = iris.data
y_true = iris.target  # Solo para evaluación, no se usa en clustering

# Crear DataFrame para análisis
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y_true
df['species_name'] = [iris.target_names[i] for i in y_true]

print(f"Forma del dataset: {X.shape}")
print(f"Características: {iris.feature_names}")
print(f"Especies (ground truth): {iris.target_names}")
print(f"Distribución de especies:")
print(df['species_name'].value_counts())

# Estadísticas descriptivas
print(f"\nEstadísticas descriptivas:")
print(df[iris.feature_names].describe())

# 2. PREPROCESAMIENTO
print("\n2. PREPROCESAMIENTO...")
# Estandarización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Datos estandarizados - Media: {X_scaled.mean(axis=0)}")
print(f"Datos estandarizados - Std: {X_scaled.std(axis=0)}")

# 3. DETERMINACIÓN DEL NÚMERO ÓPTIMO DE CLUSTERS
print("\n3. DETERMINACIÓN DEL NÚMERO ÓPTIMO DE CLUSTERS...")

# Método del codo (Elbow method)
k_range = range(1, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    
    if k > 1:  # Silhouette score requiere al menos 2 clusters
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

print(f"Inercias por k: {inertias}")
print(f"Silhouette scores por k: {silhouette_scores}")

# Encontrar k óptimo por silhouette
optimal_k_silhouette = np.argmax(silhouette_scores[1:]) + 2  # +2 porque empezamos desde k=2
print(f"K óptimo por Silhouette Score: {optimal_k_silhouette}")

# 4. APLICAR K-MEANS CON K=3 (CONOCEMOS QUE HAY 3 ESPECIES)
print("\n4. APLICANDO K-MEANS CON K=3...")
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans_3.fit_predict(X_scaled)

# Métricas de evaluación
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
ari = adjusted_rand_score(y_true, cluster_labels)

print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
print(f"Adjusted Rand Index (vs ground truth): {ari:.4f}")

print(f"\nCentroides de los clusters:")
centroids = scaler.inverse_transform(kmeans_3.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=iris.feature_names)
print(centroids_df)

# Análisis de la asignación de clusters
print(f"\nDistribución de clusters:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} puntos")

# 5. ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)
print("\n5. APLICANDO PCA...")

# PCA a 2 componentes
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

print(f"Varianza explicada por PC1: {pca_2d.explained_variance_ratio_[0]:.4f}")
print(f"Varianza explicada por PC2: {pca_2d.explained_variance_ratio_[1]:.4f}")
print(f"Varianza total explicada (2D): {pca_2d.explained_variance_ratio_.sum():.4f}")

# PCA a 3 componentes
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

print(f"Varianza explicada por PC3: {pca_3d.explained_variance_ratio_[2]:.4f}")
print(f"Varianza total explicada (3D): {pca_3d.explained_variance_ratio_.sum():.4f}")

# Componentes principales
print(f"\nComponentes principales (2D):")
components_df_2d = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1', 'PC2'],
    index=iris.feature_names
)
print(components_df_2d)

# 6. COMPARACIÓN CON GROUND TRUTH
print("\n6. COMPARACIÓN CON GROUND TRUTH...")
# Crear tabla de comparación
comparison_df = pd.DataFrame({
    'Ground Truth': y_true,
    'Species': [iris.target_names[i] for i in y_true],
    'K-Means Cluster': cluster_labels
})

print(f"Tabla de confusión (Ground Truth vs K-Means):")
confusion_table = pd.crosstab(
    comparison_df['Species'], 
    comparison_df['K-Means Cluster'],
    margins=True
)
print(confusion_table)

# 7. VISUALIZACIONES
plt.figure(figsize=(20, 15))

# Subplot 1: Método del codo
plt.subplot(3, 4, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.grid(True)

# Subplot 2: Silhouette Score
plt.subplot(3, 4, 2)
plt.plot(range(2, 11), silhouette_scores[1:], 'ro-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score por k')
plt.grid(True)

# Subplot 3: PCA 2D - Ground Truth
plt.subplot(3, 4, 3)
colors = ['red', 'blue', 'green']
for i, species in enumerate(iris.target_names):
    mask = y_true == i
    plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
               c=colors[i], label=species, alpha=0.7)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f})')
plt.title('PCA 2D - Ground Truth')
plt.legend()
plt.grid(True)

# Subplot 4: PCA 2D - K-Means Clusters
plt.subplot(3, 4, 4)
colors_cluster = ['purple', 'orange', 'cyan']
for i in range(3):
    mask = cluster_labels == i
    plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
               c=colors_cluster[i], label=f'Cluster {i}', alpha=0.7)
# Añadir centroides
centroids_pca = pca_2d.transform(kmeans_3.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           c='black', marker='x', s=200, linewidths=3, label='Centroides')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f})')
plt.title('PCA 2D - K-Means Clusters')
plt.legend()
plt.grid(True)

# Subplot 5: Características originales - Pairplot style
plt.subplot(3, 4, 5)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Sepal Length vs Sepal Width')
plt.colorbar()

# Subplot 6: Otra combinación de características
plt.subplot(3, 4, 6)
plt.scatter(X[:, 2], X[:, 3], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.title('Petal Length vs Petal Width')
plt.colorbar()

# Subplot 7: Histograma de Silhouette scores por muestra
plt.subplot(3, 4, 7)
from sklearn.metrics import silhouette_samples
sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
plt.hist(sample_silhouette_values, bins=20, alpha=0.7)
plt.axvline(silhouette_avg, color='red', linestyle='--', 
           label=f'Promedio: {silhouette_avg:.3f}')
plt.xlabel('Silhouette Score')
plt.ylabel('Frecuencia')
plt.title('Distribución de Silhouette Scores')
plt.legend()

# Subplot 8: Varianza explicada acumulada
plt.subplot(3, 4, 8)
pca_full = PCA()
pca_full.fit(X_scaled)
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cumsum_var)+1), cumsum_var, 'go-')
plt.axhline(y=0.95, color='red', linestyle='--', label='95% varianza')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada por PCA')
plt.legend()
plt.grid(True)

# Subplot 9: Biplot PCA
plt.subplot(3, 4, 9)
# Puntos
for i, species in enumerate(iris.target_names):
    mask = y_true == i
    plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
               c=colors[i], label=species, alpha=0.6)

# Vectores de características
scale_factor = 3
for i, feature in enumerate(iris.feature_names):
    plt.arrow(0, 0, 
             pca_2d.components_[0, i] * scale_factor,
             pca_2d.components_[1, i] * scale_factor,
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    plt.text(pca_2d.components_[0, i] * scale_factor * 1.15,
             pca_2d.components_[1, i] * scale_factor * 1.15,
             feature.replace(' ', '\n'), fontsize=8, ha='center')

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f})')
plt.title('PCA Biplot')
plt.legend()
plt.grid(True)

# Subplot 10: Matriz de correlación original
plt.subplot(3, 4, 10)
corr_matrix = df[iris.feature_names].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, cbar_kws={'label': 'Correlación'})
plt.title('Matriz de Correlación - Datos Originales')

# Subplot 11: Comparación de métricas por k
plt.subplot(3, 4, 11)
k_metrics = []
for k in range(2, 8):
    kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = kmeans_k.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels_k)
    ch_score = calinski_harabasz_score(X_scaled, labels_k)
    k_metrics.append([k, sil_score, ch_score/1000])  # Escalar CH para visualización

k_metrics = np.array(k_metrics)
plt.plot(k_metrics[:, 0], k_metrics[:, 1], 'o-', label='Silhouette')
plt.plot(k_metrics[:, 0], k_metrics[:, 2], 's-', label='Calinski-Harabasz/1000')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Score')
plt.title('Métricas de Validación por k')
plt.legend()
plt.grid(True)

# Subplot 12: Boxplot de características por cluster
plt.subplot(3, 4, 12)
cluster_df = df.copy()
cluster_df['Cluster'] = cluster_labels
feature_to_plot = iris.feature_names[2]  # Petal length
boxplot_data = [cluster_df[cluster_df['Cluster'] == i][feature_to_plot] 
                for i in range(3)]
plt.boxplot(boxplot_data, labels=[f'C{i}' for i in range(3)])
plt.ylabel(feature_to_plot)
plt.xlabel('Cluster')
plt.title(f'{feature_to_plot} por Cluster')

plt.tight_layout()
plt.savefig('Ejercicio3_resultados.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. VISUALIZACIÓN 3D
fig = plt.figure(figsize=(15, 5))

# PCA 3D - Ground Truth
ax1 = fig.add_subplot(131, projection='3d')
for i, species in enumerate(iris.target_names):
    mask = y_true == i
    ax1.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
               c=colors[i], label=species, alpha=0.7)
ax1.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.3f})')
ax1.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.3f})')
ax1.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.3f})')
ax1.set_title('PCA 3D - Ground Truth')
ax1.legend()

# PCA 3D - K-Means
ax2 = fig.add_subplot(132, projection='3d')
for i in range(3):
    mask = cluster_labels == i
    ax2.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
               c=colors_cluster[i], label=f'Cluster {i}', alpha=0.7)
# Centroides en 3D
centroids_pca_3d = pca_3d.transform(kmeans_3.cluster_centers_)
ax2.scatter(centroids_pca_3d[:, 0], centroids_pca_3d[:, 1], centroids_pca_3d[:, 2],
           c='black', marker='x', s=200, linewidths=3, label='Centroides')
ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.3f})')
ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.3f})')
ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.3f})')
ax2.set_title('PCA 3D - K-Means Clusters')
ax2.legend()

# Espacio original 3D (usando 3 de las 4 características)
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_labels, cmap='viridis', alpha=0.7)
ax3.set_xlabel(iris.feature_names[0])
ax3.set_ylabel(iris.feature_names[1])
ax3.set_zlabel(iris.feature_names[2])
ax3.set_title('Espacio Original 3D')

plt.tight_layout()
plt.savefig('Ejercicio3_3d.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("EJERCICIO 3 COMPLETADO")
print("="*60)
print(f"Silhouette Score (k=3): {silhouette_avg:.4f}")
print(f"Varianza explicada PCA 2D: {pca_2d.explained_variance_ratio_.sum():.4f}")
print(f"Varianza explicada PCA 3D: {pca_3d.explained_variance_ratio_.sum():.4f}")
print(f"Adjusted Rand Index: {ari:.4f}")
print("Gráficos guardados como: Ejercicio3_resultados.png y Ejercicio3_3d.png")