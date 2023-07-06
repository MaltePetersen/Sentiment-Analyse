# Sentiment Analyse zu Text Generierung
In diese Analyse wollen wir simplen bis state of the art Ansätzen zu Sentiment Analyse gehen bis hin zur Word Generation. 
1. Bag of Words Sentiment Analyse von McDonalds Reviews
2. Bag of Words Sentiment Analyse von McDonalds Reviews und Imdb Datensätzen
3. Sentiment Analyse durch Wordembeddings und LSTM Netze 
    - Zusaetzliche Nutzung von vor-trainierten embeddings ([GloVe](https://www.kaggle.com/datasets/anindya2906/glove6b))
4. Sentiment Analyse durch Transformers
5. Wordgenerierung neuer Reviews durch einen Transformer

# Overview
- McDonalds Dataset - Bag of Words
    - MultiClass Classficiation => Sentiment Analysis
- McDonalds + IMDB Dataset (Merged) - Bag of Words
    - Binary Classification - positive / negative
- McDonalds + IMDB Dataset (Merged) - LSTMs
    - Binary Classification - Sentiment Analysis
    - LSTM + GloVe Embeddings
- McDonalds + IMDB Dataset (Merged) - Transformers
    - Binary Classification - Sentiment Analysis
    - (Transformer + GloVe Embeddings)
    - Transformer + BERT
- (Review Generation from McDonalds / IMDB Dataset)
# Roter Faden
- Idee: Sentiment Analysis of Reviews
- Goal: Having a generalized Model, which can get the sentiment of any review
- TODO Readme + Explanation of models/steps taken (Malte)
- TODO Data Cleaning (Phillip)
- TODO Data Visualization in all Notebooks (Phillip)
- TODO Getting Transfomers to work with (GloVe) + BERT (Malte + Phillip)
    - General Knowledge about Attention Mechanism + Why Transformers
# Data Cleaning
- Lowercase 
- Punctuation
- HTML tags
- (Stemming)
- (Lemmatization)
# Dataset Visualisation
- Distribution
- Etc.
# Hyperparameter Tuning / Training Optimizations
- TODO Callbacks (Malte + Phillip)
- TODO Research Validation Split, when do use how much => what does this change? (Malte + Phillip)
# Model Performance Evaluation
- Training / val loss
- Training / val accuracy
- Classficiation Report (`plot_utils.py`)
- (TensorBoard)
