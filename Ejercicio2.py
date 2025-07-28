"""
Ejercicio 2: Clasificación de texto multiclase con TF-IDF y Naive Bayes
Dataset: 20 Newsgroups
Objetivo: Clasificar documentos de texto en múltiples categorías
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import string
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("EJERCICIO 2: CLASIFICACIÓN TEXTO MULTICLASE CON TF-IDF")
print("="*60)

# 1. CARGAR DATOS
print("\n1. CARGANDO DATOS...")

# Seleccionar un subconjunto de categorías para el ejemplo
categories = ['comp.graphics', 'sci.space', 'talk.politics.misc', 'rec.sport.baseball']

# Cargar datos de entrenamiento y prueba
train_data = fetch_20newsgroups(subset='train', categories=categories, 
                               shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', categories=categories,
                              shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))

print(f"Categorías seleccionadas: {categories}")
print(f"Documentos de entrenamiento: {len(train_data.data)}")
print(f"Documentos de prueba: {len(test_data.data)}")

# Mostrar distribución de clases
print(f"\nDistribución de clases en entrenamiento:")
train_counts = pd.Series(train_data.target).value_counts().sort_index()
for i, count in enumerate(train_counts):
    print(f"  {categories[i]}: {count} documentos ({count/len(train_data.data)*100:.1f}%)")

# 2. PREPROCESAMIENTO DE TEXTO
print("\n2. PREPROCESAMIENTO DE TEXTO...")

def preprocess_text(text):
    """Función de preprocesamiento básico de texto"""
    # Convertir a minúsculas
    text = text.lower()
    # Remover números
    text = re.sub(r'\d+', '', text)
    # Remover puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remover espacios extra
    text = ' '.join(text.split())
    return text

# Aplicar preprocesamiento
print("Aplicando preprocesamiento...")
train_texts_processed = [preprocess_text(text) for text in train_data.data]
test_texts_processed = [preprocess_text(text) for text in test_data.data]

# Mostrar ejemplo de preprocesamiento
print(f"\nEjemplo de texto original:")
print(train_data.data[0][:200] + "...")
print(f"\nEjemplo de texto preprocesado:")
print(train_texts_processed[0][:200] + "...")

# 3. REPRESENTACIÓN TF-IDF
print("\n3. CREANDO REPRESENTACIÓN TF-IDF...")

# Crear vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,  # Limitar a las 10000 palabras más frecuentes
    stop_words='english',  # Remover palabras vacías en inglés
    ngram_range=(1, 2),  # Usar unigramas y bigramas
    min_df=2,  # Palabra debe aparecer al menos en 2 documentos
    max_df=0.95  # Ignorar palabras que aparecen en más del 95% de documentos
)

# Ajustar y transformar datos de entrenamiento
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts_processed)
X_test_tfidf = tfidf_vectorizer.transform(test_texts_processed)

print(f"Forma de matriz TF-IDF entrenamiento: {X_train_tfidf.shape}")
print(f"Forma de matriz TF-IDF prueba: {X_test_tfidf.shape}")
print(f"Número de características (vocabulario): {len(tfidf_vectorizer.vocabulary_)}")

# Mostrar algunas características más importantes
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"\nPrimeras 20 características del vocabulario:")
print(feature_names[:20])

# 4. ENTRENAMIENTO DEL MODELO
print("\n4. ENTRENAMIENTO NAIVE BAYES MULTINOMIAL...")

# Crear y entrenar el modelo
nb_classifier = MultinomialNB(alpha=1.0)  # alpha=1.0 es suavizado de Laplace
nb_classifier.fit(X_train_tfidf, train_data.target)

# Hacer predicciones
y_pred = nb_classifier.predict(X_test_tfidf)
y_true = test_data.target

# 5. EVALUACIÓN DEL MODELO
print("\n5. EVALUACIÓN DEL MODELO...")

# Accuracy general
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Reporte de clasificación detallado
print(f"\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=categories))

# Matriz de confusión
print(f"\nMatriz de Confusión:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# 6. ANÁLISIS DE CARACTERÍSTICAS IMPORTANTES
print("\n6. ANÁLISIS DE CARACTERÍSTICAS...")

# Obtener las características más importantes para cada clase
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
for i, category in enumerate(categories):
    # Obtener los coeficientes log-probabilidad para esta clase
    log_prob = nb_classifier.feature_log_prob_[i]
    # Obtener las top 10 características
    top_features_idx = np.argsort(log_prob)[-10:]
    top_features = feature_names[top_features_idx]
    print(f"\nTop 10 características para '{category}':")
    print(top_features[::-1])  # Mostrar en orden descendente

# 7. EJEMPLOS DE PREDICCIONES
print("\n7. EJEMPLOS DE PREDICCIONES...")

# Mostrar algunas predicciones específicas
for i in range(5):
    actual_category = categories[y_true[i]]
    predicted_category = categories[y_pred[i]]
    confidence = nb_classifier.predict_proba(X_test_tfidf[i:i+1])[0]
    max_confidence = confidence.max()
    
    print(f"\nEjemplo {i+1}:")
    print(f"Texto: {test_data.data[i][:100]}...")
    print(f"Categoría real: {actual_category}")
    print(f"Categoría predicha: {predicted_category}")
    print(f"Confianza: {max_confidence:.4f}")
    print(f"Correcto: {'✓' if actual_category == predicted_category else '✗'}")

# 8. VISUALIZACIONES
plt.figure(figsize=(15, 10))

# Subplot 1: Matriz de confusión
plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[cat.replace('.', '\n') for cat in categories],
            yticklabels=[cat.replace('.', '\n') for cat in categories])
plt.title('Matriz de Confusión')
plt.ylabel('Categoría Real')
plt.xlabel('Categoría Predicha')

# Subplot 2: Distribución de clases
plt.subplot(2, 3, 2)
class_counts = pd.Series(train_data.target).value_counts().sort_index()
plt.bar([cat.replace('.', '\n') for cat in categories], class_counts.values)
plt.title('Distribución de Clases (Entrenamiento)')
plt.ylabel('Número de Documentos')
plt.xticks(rotation=45)

# Subplot 3: Accuracy por clase
plt.subplot(2, 3, 3)
# Calcular accuracy por clase
class_accuracies = []
for i in range(len(categories)):
    class_mask = (y_true == i)
    if class_mask.sum() > 0:
        class_acc = (y_pred[class_mask] == y_true[class_mask]).mean()
        class_accuracies.append(class_acc)
    else:
        class_accuracies.append(0)

plt.bar([cat.replace('.', '\n') for cat in categories], class_accuracies)
plt.title('Accuracy por Clase')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim([0, 1])

# Subplot 4: Top palabras globales TF-IDF
plt.subplot(2, 3, 4)
# Calcular TF-IDF promedio por característica
mean_tfidf = np.array(X_train_tfidf.mean(axis=0)).flatten()
top_global_idx = np.argsort(mean_tfidf)[-15:]
top_global_words = feature_names[top_global_idx]
top_global_scores = mean_tfidf[top_global_idx]

plt.barh(range(15), top_global_scores)
plt.yticks(range(15), top_global_words)
plt.xlabel('TF-IDF Promedio')
plt.title('Top 15 Palabras por TF-IDF')

# Subplot 5: Distribución de longitud de documentos
plt.subplot(2, 3, 5)
doc_lengths = [len(doc.split()) for doc in train_texts_processed]
plt.hist(doc_lengths, bins=50, alpha=0.7)
plt.xlabel('Longitud del Documento (palabras)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Longitud de Documentos')

# Subplot 6: Comparación de métricas
plt.subplot(2, 3, 6)
# Obtener métricas del classification_report
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

x = np.arange(len(categories))
width = 0.25

plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
plt.bar(x, recall, width, label='Recall', alpha=0.8)
plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

plt.xlabel('Categorías')
plt.ylabel('Score')
plt.title('Métricas por Categoría')
plt.xticks(x, [cat.replace('.', '\n') for cat in categories])
plt.legend()
plt.ylim([0, 1])

plt.tight_layout()
plt.savefig('Ejercicio2_resultados.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. PRUEBA CON DIFERENTES PARÁMETROS DE SUAVIZADO
print("\n9. EFECTO DEL SUAVIZADO (ALPHA)...")
alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
alpha_results = []

for alpha in alphas:
    nb_alpha = MultinomialNB(alpha=alpha)
    nb_alpha.fit(X_train_tfidf, train_data.target)
    y_pred_alpha = nb_alpha.predict(X_test_tfidf)
    acc_alpha = accuracy_score(y_true, y_pred_alpha)
    alpha_results.append(acc_alpha)
    print(f"Alpha = {alpha}: Accuracy = {acc_alpha:.4f}")

print(f"\nMejor alpha: {alphas[np.argmax(alpha_results)]} con accuracy: {max(alpha_results):.4f}")

print("\n" + "="*60)
print("EJERCICIO 2 COMPLETADO")
print("="*60)
print(f"Accuracy final: {accuracy:.4f}")
print(f"Número de categorías: {len(categories)}")
print(f"Características TF-IDF: {X_train_tfidf.shape[1]}")
print("Gráfico guardado como: Ejercicio2_resultados.png")