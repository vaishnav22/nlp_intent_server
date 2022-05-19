import nltk
#importing PorterStemmer from nltk stemmer
from nltk.stem.porter import PorterStemmer
import numpy as np

p_stemmer = PorterStemmer()

def tok_sentance(sentance):
    return nltk.word_tokenize(sentance)
  
def stem_word(word):
    return p_stemmer.stem(word.lower())

def b_o_w(sentace_tok,all_words):
    sentace_tok = [stem_word(w) for w in sentace_tok]

#creating bag of word and assigning 1.0 to the word present in the bag of word
    word_b = np.zeros(len(all_words), dtype=np.float32)
    for ind, word in enumerate(all_words):
        if word in sentace_tok:
            word_b[ind] = 1.0 #assgning word as 1.0

#return the bag of word
    return word_b