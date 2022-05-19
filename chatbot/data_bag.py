import nltk
import numpy as np
import json

from chatbot.nltk_utils import tok_sentance, stem_word, b_o_w


nltk.download('punkt')

pointer = open('data_intents.json')

intent_file = json.load(pointer)

# creating bag of all words store the bag of words 
bag_all_of_words = []
# creating a list to only append the tags from the intents.json file as seen above
intents_tags = []
# taging each word with its "tag"
word_tag= []

for intent in intent_file['data']:
    tag = intent['tagged_data']
    intents_tags.append(tag)
    for sentance in intent['user_input']:
        word = tok_sentance(sentance)
        bag_all_of_words.extend(word)
        word_tag.append((word, tag))

# ignore these special character
ignore_words = ['?','!','.',',','/','\\']
# creating the bag of all workds using the bag of words and stemming them 
bag_all_of_words = [stem_word(w) for w in bag_all_of_words if w not in ignore_words]
bag_all_of_words = sorted(set(bag_all_of_words))
tags = sorted(set(intents_tags))