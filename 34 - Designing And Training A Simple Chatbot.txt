import pandas as pd
import nltk 
data=pd.read_excel('ml ds/training dataset.xlsx')
#data preprocessing
nltk.download('punkt')
data['Concept']=data['Concept'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))
print(data.head())


#vectorise text data
#Convert text data into numerical using TF-IDF (term frequency-inverse document frequency)VECTORIZATION.


from sklearn.feature_extraction.text import TfidfVectorizer
vector=TfidfVectorizer()
X=vector.fit_transform(data['Concept'])
print(X.shape)

#train a text classification model
#Train a text classification model using the vectorised data and the labels.

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

#split
x_train,x_test,y_train,y_test= train_test_split(data['Concept'], data['Description'], test_size=0.2, random_state=42)

#create model pipeline
model= make_pipeline(TfidfVectorizer(),MultinomialNB() )

#training
model.fit(x_train,y_train)
print("completed training")


#implement a function to get respond from chat bot
def get_response(question):
    question = ' '.join(nltk.word_tokenize(question.lower()))
    answer = model.predict([question])[0]
    return answer

# Testing
print(get_response("What is machine learning?"))