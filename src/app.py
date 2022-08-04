import re
import pandas as pd
import pickle
import numpy as np
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')
df_raw

df_raw.info()
df_raw.describe()
df_raw.sample(20)

# invalanced
df_raw['is_spam'].value_counts()

df_interin = df_raw.copy()
print(f'Duplicated rows: {df_interin.duplicated().sum()}')

df_interin = df_interin.drop_duplicates().reset_index(drop=True)
df_interin.info()

# continue invalanced?
df_interin['is_spam'].value_counts()

# functions to clean the text
def comas(text):
    """
    Elimina comas del texto
    """
    return re.sub(',', ' ', text)

def espacios(text):
    """
    Elimina enters dobles por un solo enter
    """
    return re.sub(r'(\n{2,})','\n', text)

def minuscula(text):
    """
    Cambia mayusculas a minusculas
    """
    return text.lower()

def numeros(text):
    """
    Sustituye los numeros
    """
    return re.sub('([\d]+)', ' ', text)

def caracteres_no_alfanumericos(text):
    """
    Sustituye caracteres raros, no digitos y letras
    Ej. hola 'pepito' como le va? -> hola pepito como le va
    """
    return re.sub("(\\W)+"," ",text)

def comillas(text):
    """
    Sustituye comillas por un espacio
    Ej. hola 'pepito' como le va? -> hola pepito como le va?
    """
    return re.sub("'"," ", text)

def palabras_repetidas(text):
    """
    Sustituye palabras repetidas

    Ej. hola hola, como les va? a a ustedes -> hola, como les va? a ustedes
    """
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

def esp_multiple(text):
    """
    Sustituye los espacios dobles entre palabras
    """
    return re.sub(' +', ' ',text)
    
def url(text):
    """
    Remove https
    """
    return re.sub(r'(https://www|https://)', '', text)

df_interin['clean_url'] = df_interin['url'].apply(url).apply(caracteres_no_alfanumericos).apply(esp_multiple)
df_interin.head()
df_interin['is_spam'] = df_interin['is_spam'].astype(int)
df_interin

df = df_interin.copy()

X = df['clean_url']
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

X_train.shape
X_test.shape

classifier = SVC(C = 1.0, kernel = 'linear', gamma = 'auto')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))

param_grid = {
    'C': [0.1,1, 10, 100], 
    'gamma': [1,0.1,0.01,0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}
grid = GridSearchCV(SVC(random_state=42), param_grid,verbose=2)
grid.fit(X_train,y_train)

grid.best_params_
grid.best_estimator_

pred_grid = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, pred_grid))

best_model = classifier
pickle.dump(best_model, open('../models/best_model.pickle', 'wb')) 


