import io
import numpy as np

"""
This function writes the word embeddings to a file. Can be used by https://projector.tensorflow.org/
"""


def write_embeddings(vocab_size, reverse_word_index, embedding_weights, vecs_filename='vecs.tsv', meta_filename='meta.tsv'):
    out_v = io.open(vecs_filename, 'w', encoding='utf-8')
    out_m = io.open(meta_filename, 'w', encoding='utf-8')

    # Initialize the loop. Start counting at `1` because `0` is just for the padding
    for word_num in range(1, vocab_size):

        # Get the word associated at the current index
        word_name = reverse_word_index[word_num]

        # Get the embedding weights associated with the current index
        word_embedding = embedding_weights[word_num]

        # Write the word name
        out_m.write(word_name + "\n")

        # Write the word embedding
        out_v.write('\t'.join([str(x) for x in word_embedding]) + "\n")

    # Close the files
    out_v.close()
    out_m.close()


def read_embeddings(filepath):
    embeddings = {}
    # Read word vectors
    with open(filepath) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
    return embeddings
