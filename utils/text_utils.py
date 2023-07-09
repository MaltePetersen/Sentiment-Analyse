import re
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def clean_text(text):
    # Convert to lowercase
    test = text.lower()
    # Remove punctuation
    test = re.sub(r'[^a-z\s]', '', test)
    # Remove HTML Tags
    test = re.sub(r'<.*?>', '', test)
    # Remove stopwords
    test = ' '.join([word for word in word_tokenize(test) if word not in stopwords])
    return test