import sys
import nltk
nltk.download(['stopwords','punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
import pickle

def load_data(database_filepath):
    '''
    Function to load the sql database using the database filepath and split the dataset features X dataset and target Y dataset.
    Returns X and Y datasets alongwith the category names
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster_Combined', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
           
    return X, Y, category_names

def tokenize(text):
   '''
   Function to tokenize the text messages.
   Inputs the complete text messages and returns clean tokenized text as a list
   '''

   # Normalize
   # Set text to lower case and remove punctuation
   text= text.lower()
   text = re.sub(r"[^a-zA-Z0-9]", " ", text)
   # Tokenize words 
   tokens = word_tokenize(text)
   # lemmatizer and remove stopwords
   # lemmatizer
   lemmatizer = WordNetLemmatizer()
   # stopwords
   stop_words = set(stopwords.words('english'))
   # lemmatizer and remove stopwords
   clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if not w in stop_words]
   clean_tokens = [] 
   for w in tokens: 
       if w not in stop_words: 
           clean_tokens.append(w)
   return clean_tokens

def get_results(y_test, y_pred):
    results = pd.DataFrame(columns=['Category', 'f1_score', 'precision', 'recall'])
    num = 0
    for cat in y_test.columns:
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test[cat], y_pred[:,num], average='weighted')
        results.set_value(num+1, 'Category', cat)
        results.set_value(num+1, 'f1_score', f1_score)
        results.set_value(num+1, 'precision', precision)
        results.set_value(num+1, 'recall', recall)
        num += 1
    print('f1_score:', results['f1_score'].mean())
    print('precision:', results['precision'].mean())
    print('recall:', results['recall'].mean())
    return results


def performance_metric(y_true, y_pred):
    """Calculate median F1 score for all of the output classifiers
    
    Args:
    y_true: array. Array containing actual labels.
    y_pred: array. Array containing predicted labels.
        
    Returns:
    score: float. Median F1 score for all of the output classifiers
    """
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i],average='micro')
        f1_list.append(f1)
        
    score = np.median(f1_list)
    return score

def build_model():
    '''
    Function to build the model by creating an ML pipeline and use a set of parameters to further tune the model by applying Grid Search
    '''
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
              'clf__estimator__n_estimators':[10, 25]
             }

   
    scorer = make_scorer(performance_metric)
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, verbose = 10)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to test the model for each category using accuracy scores
    '''
    y_pred = model.predict(X_test)
    # print classification report
    for i, col in enumerate(category_names):
        print (col)
        print(classification_report(Y_test[col], y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Function to save the model
    '''
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()