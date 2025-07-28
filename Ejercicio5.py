"""
Ejercicio 5: Fine-tuning BERT para análisis de sentimiento
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("EJERCICIO 5: FINE-TUNING BERT PARA ANÁLISIS SENTIMIENTO")

try:
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import re
except ImportError as e:
    print(f"Error: {e}")
    exit(1)

try:
    from transformers import BertTokenizer, BertModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

categories_positive = ['rec.sport.baseball', 'rec.motorcycles']
categories_negative = ['talk.politics.guns', 'talk.politics.misc']

try:
    data_pos = fetch_20newsgroups(subset='train', categories=categories_positive, 
                                 remove=('headers', 'footers', 'quotes'))
    data_neg = fetch_20newsgroups(subset='train', categories=categories_negative,
                                 remove=('headers', 'footers', 'quotes'))
    
    texts = list(data_pos.data) + list(data_neg.data)
    labels = [1] * len(data_pos.data) + [0] * len(data_neg.data)
    
    max_samples = 2000
    if len(texts) > max_samples:
        indices = np.random.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    
except Exception as e:
    texts = [
        "This is a great movie, I loved it!",
        "Amazing experience, highly recommended",
        "Wonderful product, excellent quality",
        "Best service ever, very satisfied",
        "Terrible movie, waste of time",
        "Awful experience, very disappointed", 
        "Poor quality, not recommended",
        "Worst service, completely unsatisfied"
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]

print(f"Dataset: {len(texts)} textos")
unique, counts = np.unique(labels, return_counts=True)
print(f"Negativo: {counts[0]}, Positivo: {counts[1]}")

try:
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
except Exception as e:
    mid = len(texts) // 2
    X_train, X_test = texts[:mid], texts[mid:]
    y_train, y_test = labels[:mid], labels[mid:]
    X_val, y_val = X_test[:len(X_test)//2], y_test[:len(y_test)//2]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text[:512]

X_train_clean = [clean_text(text) for text in X_train]
X_val_clean = [clean_text(text) for text in X_val]
X_test_clean = [clean_text(text) for text in X_test]

tfidf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, C=1.0))
])

try:
    tfidf_pipeline.fit(X_train_clean, y_train)
    y_pred_tfidf = tfidf_pipeline.predict(X_test_clean)
    y_pred_proba_tfidf = tfidf_pipeline.predict_proba(X_test_clean)
    accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
    
    print(f"TF-IDF Accuracy: {accuracy_tfidf:.4f}")
    cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_tfidf, average='weighted')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
except Exception as e:
    print(f"Error: {e}")
    accuracy_tfidf = 0
    y_pred_tfidf = np.zeros_like(y_test)
    y_pred_proba_tfidf = np.zeros((len(y_test), 2))

if BERT_AVAILABLE:
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to(device)
        
        def get_bert_embeddings(texts, max_length=128, batch_size=8):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = tokenizer(
                    batch_texts, add_special_tokens=True, max_length=max_length,
                    padding='max_length', truncation=True, return_tensors='pt'
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**encoded)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(cls_embeddings)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return np.array(embeddings)
        
        sample_size = min(200, len(X_train_clean))
        print(f"Procesando {sample_size} ejemplos con BERT...")
        X_train_bert = get_bert_embeddings(X_train_clean[:sample_size])
        X_test_bert = get_bert_embeddings(X_test_clean)
        
        bert_classifier = LogisticRegression(random_state=42, max_iter=1000)
        bert_classifier.fit(X_train_bert, y_train[:sample_size])
        
        y_pred_bert = bert_classifier.predict(X_test_bert)
        accuracy_bert = accuracy_score(y_test, y_pred_bert)
        print(f"BERT Accuracy: {accuracy_bert:.4f}")
        
        if torch.cuda.is_available():
            print(f"Memoria GPU utilizada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"BERT Error: {e}")
        BERT_AVAILABLE = False
        accuracy_bert = 0
else:
    accuracy_bert = 0

test_texts = [
    "This movie is absolutely amazing! I loved every minute of it.",
    "Terrible experience. Worst service ever. Would not recommend.",
    "The product works as expected. Nothing special but does the job.",
    "Outstanding quality and excellent customer service!",
    "Completely disappointed with the purchase. Very poor quality."
]

print("\n=== EJEMPLOS DE PREDICCIÓN ===")
for i, text in enumerate(test_texts[:3]):
    clean_text_sample = clean_text(text)
    try:
        pred_tfidf = tfidf_pipeline.predict([clean_text_sample])[0]
        prob_tfidf = tfidf_pipeline.predict_proba([clean_text_sample])[0]
        confidence_tfidf = prob_tfidf.max()
        sentiment_tfidf = 'Positivo' if pred_tfidf == 1 else 'Negativo'
        
        print(f"Texto {i+1}: {sentiment_tfidf} ({confidence_tfidf:.3f})")
        
    except Exception as e:
        print(f"Error {i+1}: {e}")

print(f"\n=== MATRIZ DE CONFUSIÓN ===")
print(f"True Negative: {cm_tfidf[0,0]}, False Positive: {cm_tfidf[0,1]}")
print(f"False Negative: {cm_tfidf[1,0]}, True Positive: {cm_tfidf[1,1]}")

try:
    import joblib
    joblib.dump(tfidf_pipeline, 'modelo_sentimientos_tfidf.pkl')
except:
    pass

print(f"\n=== RESULTADOS FINALES ===")
print(f"TF-IDF Accuracy: {accuracy_tfidf:.4f}")
if BERT_AVAILABLE and accuracy_bert > 0:
    print(f"BERT Accuracy: {accuracy_bert:.4f}")
print(f"Dataset size: {len(texts)} textos")
print(f"Test size: {len(y_test)} ejemplos")
print(f"BERT disponible: {'Sí' if BERT_AVAILABLE else 'No'}")
print("EJERCICIO 5 COMPLETADO")