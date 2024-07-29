import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# 1. Tokenizing
from nltk.tokenize import word_tokenize

text = "NLP is quite fascinating"
tokens = word_tokenize(text)
print("Tokens:", tokens)

# 2. Stemming
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ['eating', 'eats', 'ate']
stems = [stemmer.stem(word) for word in words]
print("Stems:", stems)

# Additional words for stemming example
wordss = ["running", "ran", "runs"]
stems_additional = [stemmer.stem(word) for word in wordss]
print("Additional Stems:", stems_additional)

# 3. Lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word, pos='v') for word in words]
print("Lemmas:", lemmas)

# 4. Stop words removal
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
filtered_text = [word for word in tokens if word.lower() not in stop_words]
print("Filtered Text:", filtered_text)
