import re
from nltk.corpus import stopwords
import contractions
from collections import Counter
import nltk

stopwords = set(stopwords.words('english'))

def clean_text(text, disable_stopwords=False):
    """
    Clean text by removing unnecessary characters and altering the format of words.

    Args:
        text (str): The string of text to be cleaned.
        disable_stopwords (bool): Whether to remove stopwords or not
    
    Returns:
        str: The cleaned text.
    """

    # Convert to lowercase
    tmp = text.lower()
    
    # Expand contractions return string
    tmp = str(contractions.fix(tmp))

    # Remove non-ASCII characters
    tmp = tmp.encode('ascii', 'ignore').decode('ascii')

    # Remove HTML Tags with regex
    tmp = re.sub(r'<.*?>', '', tmp)

    # Remove URLs
    tmp = re.sub(r'http\S+', '', tmp)

    # Remove punctuation
    tmp = re.sub(r'[^\w\s]', '', tmp)


    # Remove stopwords
    # if not disable_stopwords:
        # tmp = ' '.join([word for word in word_tokenize(tmp) if word not in stopwords])

    return tmp 


def get_most_common_words(df, column, num_words=10):
    """
    Function to get the most common words in a given column of a DataFrame.
    
    Args:
    df: pandas DataFrame.
    column: str, column in df which should be analyzed.
    num_words: int, the number of most common words to return.
    
    Returns:
    most_common_words: List of tuples, the most common words and their counts.
    """

    # Split the text into words
    words = df[column].str.split(expand=True).stack()

    # Count the frequency of each word
    word_counts = Counter(words)

    # Get the most common words
    most_common_words = word_counts.most_common(num_words)
    
    return most_common_words


def get_most_common_ngrams(df, column, n=2, num_ngrams=10):
    """
    Function to get the most common n-grams in a given column of a DataFrame.
    
    Args:
    df: pandas DataFrame.
    column: str, column in df which should be analyzed.
    n: int, the size of the n-grams.
    num_ngrams: int, the number of most common n-grams to return.
    
    Returns:
    most_common_ngrams: List of tuples, the most common n-grams and their counts.
    """

    # Extract n-grams from the given column of the DataFrame, do this in memory, don't write to df
    ngrams_list = df[column].apply(lambda x: list(nltk.ngrams(x.split(), n))).tolist()

    # Flatten the list of n-grams and count the frequency of each n-gram
    ngrams = [ngram for sublist in ngrams_list for ngram in sublist]
    ngram_counts = Counter(ngrams)

    # Get the most common n-grams
    most_common_ngrams = ngram_counts.most_common(num_ngrams)
    
    return most_common_ngrams