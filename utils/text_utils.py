import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import contractions

stopwords = set(stopwords.words('english'))

def clean_text(text, disable_stopwords=False):
    # Convert to lowercase
    tmp = text.lower()
    
    # Expand contractions return string
    tmp = str(contractions.fix(tmp))

    # Remove HTML Tags
    # tmp = BeautifulSoup(str(tmp), 'html.parser').get_text()

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