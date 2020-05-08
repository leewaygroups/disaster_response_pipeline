# import libraries
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
set(stopwords.words('english'))              
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class VerbsFrequencyMetrics(BaseEstimator, TransformerMixin):

    def verbs_frequency_score(self, text):
        words = nltk.word_tokenize(text)
        
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in word_tokenize(text) if len(tok)>2]
        
        action_words = [ tag[1] for tag in nltk.pos_tag(tokens) if tag[1] in ['VB','VBD','VBG', 'VBN', 'VBP', 'VBZ']]

        return len(action_words)/(len(words)+1)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.verbs_frequency_score)
        return pd.DataFrame(X_tagged)
    
class TextSizeExtractor(BaseEstimator, TransformerMixin):
    
    def text_size(self, text):
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in word_tokenize(text) if len(tok)>2]
        
        return len(tokens)
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_size = pd.Series(X).apply(self.text_size)
        return pd.DataFrame(X_size)


def load_data(database_filepath):
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df = pd.read_sql_table(table_name='InsertTableName', con=engine)
    X = df['message'].values
    Y = df.iloc[:, 4:-1].values

    return X, Y


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in word_tokenize(text) if len(tok)>2]

    return tokens


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipleine', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            
            ('verbs_frequency_score', VerbsFrequencyMetrics())
            # ('text_size', TextSizeExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters_2 = {
        'features__text_pipleine__vect__ngram_range': ((1,1),(1,2)),
        'features__text_pipleine__vect__max_df': (0.5, 0.75, 1.0)
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    accuracy = (Y_test==y_pred).mean()*100

    print(accuracy)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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