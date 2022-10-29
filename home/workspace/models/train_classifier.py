import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    
    '''
    Function to load data from sql lite db.
    
    INPUTS:
    database_filepath: path to sqlite db file
    
    OUTPUT:
    X: message column
    Y: 36 output categories
    category_names: name of output categories
    
    '''
    # load data from database
    engine = create_engine('sqlite://///../' + database_filepath)
    df = pd.read_sql_table(con=engine,table_name='InsertTableName')
    X = df['message'].values
    Y = df.iloc[:,4:].values
    category_names = df.drop(["id", "message", "original", "genre"], axis=1).columns
    return X,Y,category_names


def tokenize(text):
    
    '''
    Function to tokenize words within a message.
    
    INPUTS:
    text: message to be word tokenized
    
    OUTPUT:
    tokens: cleaned word tokens of message
    
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r"^a-bA-B0-9"," ",text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens
    


def build_model():
    
    '''
    Function to build model pipeline with feature extraction and estimator.
    
    INPUTS:
    None
    
    OUTPUT:
    cv: built model
    
    '''
    #build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(),n_jobs=1))
    ])
    
    parameters =  {
        
         'clf__estimator__criterion': ["gini", "entropy"],
        
        'clf__estimator__n_jobs':[-1]
      
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
     '''
    Function to print out evaluation of trained model on test data.
    
    INPUTS:
    model: trained model
    X_test: messages test data
    Y_test: output categories test data
    category_names: name of output categories
    
    OUTPUT:
    None
    
    '''
    
    y_pred = model.predict(X_test)
    i=0
    for category in category_names:
        print("output category in column {}: {}".format(i, category))
        evaluation_report = classification_report(Y_test[:,i], y_pred[:,i])
        i+=1
        print(evaluation_report)
    


def save_model(model, model_filepath):
    
    ''
    Function to export model as a pickle file.
    
    INPUTS:
    model: trained model
    model_filepath: path and filename to save pickle file of trained model
    
    OUTPUT:
    None
    
    '''
    #export the mode as pickle file
    filename = model_filepath
    pickle.dump(model, open(filename, "wb"))


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
