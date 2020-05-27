import sys

import joblib
import nltk
import pandas as pd
from sqlalchemy import create_engine
import string
import re


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator,TransformerMixin

nltk.download(['punkt', 'wordnet', 'stopwords'])
stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
remove_punc_table = str.maketrans('', '', string.punctuation)

def load_data(database_filepath):
    """Loads X and Y and gets category names
    Args:
        database_filepath (str): string filepath of the sqlite database
    Returns:
        X (pandas dataframe): Feature data, just the messages
        Y (pandas dataframe): Classification labels
        category_names (list): List of the category names for classification
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    engine.dispose()

    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    Y=Y.astype(int)

 
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """Basic tokenizer that removes punctuation,numbers,html and stopwords then lemmatizes
    Args:
        text (string): input message to tokenize
    Returns:
        tokens (list): list of cleaned tokens in the message
    """
    sentence = str(text)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    rem_punct = re.sub(r"[^a-zA-Z0-9]", ' ', sentence)
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', rem_punct)
    rem_url=re.sub(r'http\S+', ' ',cleantext)
    rem_num = re.sub('[0-9]+', ' ', rem_url)
    tokens = nltk.word_tokenize(text)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stop_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w,pos='v') for w in lemma_words]
    return " ".join(lemma_words)

class LengthTransformer(BaseEstimator, TransformerMixin):
    """Class function for Pipeline object.
    Count number of word in a sentence, and add it as a new feature"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[len(x)] for x in X]

def build_model():
    """Returns the Pipeline object to be used as the model
    Args:
        None
    Returns:
        pipeline (scikit-learn Pipeline): Pipeline model object
    """
    forest_clf = RandomForestClassifier(n_estimators=10)
        
    pipeline = Pipeline([
    ('features', FeatureUnion([

        ('text_pipeline', Pipeline([
             ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
              ('SVD', TruncatedSVD(n_components=50)),
        ])),

        ('Length', LengthTransformer())
    ])),

    ('MultiOutput', MultiOutputClassifier(forest_clf))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Args:
        model (pandas dataframe): the scikit-learn fitted model
        X_text (pandas dataframe): The X test set
        Y_test (pandas dataframe): the Y test classifications
        category_names (list): the category names
    Returns:
        Global Accuracy,Precision,Recall and classification_report for each labels
    """
    Y_pred = model.predict(X_test)
    print('Performance for each labels\n')
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))



def save_model(model, model_filepath):
    """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to
    Returns:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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