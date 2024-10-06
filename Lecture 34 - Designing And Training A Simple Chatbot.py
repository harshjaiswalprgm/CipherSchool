import pandas as pd
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# File path to the Excel file
file_path = 'ml ds/training dataset.xlsx'  # Update this path as needed

# Check if the file exists, if not, create a new Excel file with default data
if not os.path.exists(file_path):
    print(f"The file '{file_path}' does not exist. Creating a new file...")

    # Create a default dataset
    data_dict = {
        'Concept': [
            'Machine Learning is a field of artificial intelligence',
            'Natural language processing is a subfield of AI',
            'Neural networks are used in deep learning',
            'Supervised learning uses labeled data',
            'Unsupervised learning uses unlabeled data'
        ],
        'Description': [
            'Machine learning refers to the process of using algorithms to analyze data and learn from it.',
            'NLP involves the interaction between computers and human language.',
            'Neural networks are a series of algorithms that mimic the operations of a human brain.',
            'Supervised learning is when a model is trained on labeled data to predict outcomes.',
            'Unsupervised learning identifies hidden patterns in unlabeled data.'
        ]
    }

    # Create a DataFrame from the dictionary
    data = pd.DataFrame(data_dict)

    # Create directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save DataFrame as an Excel file
    data.to_excel(file_path, index=False)
    print(f"New file created at '{file_path}'.")

else:
    # Read the data from the existing file
    data = pd.read_excel(file_path)

# Data preprocessing
nltk.download('punkt')
data['Concept'] = data['Concept'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))
print(data.head())  # Show the first few rows to confirm it's been read correctly

# Vectorize text data
vector = TfidfVectorizer()
X = vector.fit_transform(data['Concept'])
print(f"Vectorized text data shape: {X.shape}")

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data['Concept'], data['Description'], test_size=0.2, random_state=42)

# Create model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Training the model
model.fit(x_train, y_train)
print("Training completed successfully")

# Implement a function to get a response from the chatbot
def get_response(question):
    # Preprocess the question
    question = ' '.join(nltk.word_tokenize(question.lower()))
    # Predict the response based on the model
    answer = model.predict([question])[0]
    return answer

# Testing the chatbot
test_question = "What is machine learning?"
print(f"Question: {test_question}")
print(f"Answer: {get_response(test_question)}")
