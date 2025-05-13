import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """ split the sentence to words """
    return nltk.word_tokenize(sentence)

def stem(word):
    """ return the word to its origine """
    return stemmer.stem(word.lower())

def bag_of_words(sentence, all_words):
    
    sentence_words = [stem(word) for word in sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w in enumerate(all_words):
        if w in sentence_words:
            bag[i] = 1
            
    return bag
        
            
