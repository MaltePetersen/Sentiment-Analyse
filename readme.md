# Sentiment Analyse
In diese Repository werden simple bis State of the Art Ansätze zu Sentiment Analyse vorgestellt. 

Gliederung: 

1. Explorative Analyse der Daten
2. Bag of Words Sentiment Analyse von McDonalds Reviews
3. Bag of Words Sentiment Analyse von McDonalds Reviews und Imdb Datensätzen
4. Sentiment Analyse durch Wordembeddings und LSTM Netze 
    - Zusaetzliche Nutzung von vor-trainierten embeddings ([GloVe](https://www.kaggle.com/datasets/anindya2906/glove6b))
5. Sentiment Analyse durch Transformers


Das Repository durchläuft fünf Schritte, um von einem spezifischen Sentiment-Analyse-Modell zu einem generischen Modell zu gelangen. Im ersten Kapitel werden alle verwendeten Daten analysiert. Im zweiten Kapitel wird eine Bag-of-Words-Sentiment-Analyse von McDonald's Reviews durchgeführt. Im dritten Kapitel wird weiterhin das Bag-of-Words-Modell verwendet, jedoch wird das Datenset um IMDB-Filmkritiken erweitert und die Datenstruktur von McDonald's wird optimiert. Das Ziel besteht darin, einen ersten Schritt hin zu einem generischeren Modell zu machen. Im vierten Kapitel wird ein LSTM-Modell mit Word Embeddings auf den Daten des vorherigen Kapitels trainiert, wobei auch das vortrainierte GloVe Embedding verwendet wird. Darüber hinaus werden neue Erkenntnisse über generische Modelle gewonnen. Im letzten Kapitel wird der Transformer Encoder verwendet, um eine State-of-the-Art-Sentiment-Analyse durchzuführen, wobei einige Fragen bezüglich der Auswirkung des Positional Embeddings aufkommen.


| Analyse                                               | Accuracy | Klassen |
|-------------------------------------------------------|----------|---------|
| Bigram McDonalds Reviews                              | 45.06%   | 5       |
| Bigram McDonalds und IMDB Reviews                     | 89.19%   | 2       |
| LSTM McDonalds und IMDB Reviews                       | 88.64%   | 2       |
| LSTM McDonalds und IMDB Reviews GloVe                 | 87.53%   | 2       |
| Transformer nur Attention                             | 88.77%   | 2       |
| Transformer Attention und PostionalEnmbedding         | 84.70%   | 2       |
| Transformer Attention und simples PostionalEnmbedding | 89.72%   | 2       |


# Referenzen: 
Deep Learning with Python 2nd edition - François Chollet

Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow - Aurélien Géron

[Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings](https://www.youtube.com/watch?v=dichIcUZfOw)

[The Transformer Positional Encoding Layer in Keras, Part 2](https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/)
