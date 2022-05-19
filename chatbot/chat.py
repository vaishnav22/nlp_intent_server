import random
import json
import torch
from chatbot.nltk_utils import tok_sentance, stem_word, b_o_w
from chatbot.model import NeruralNet

from chatbot.data_bag import *

def chatbot(user_input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pointer = open('data_intents.json')

    intent_file = json.load(pointer)



    model_file = "model_pickled_adam.pth"
    data_json = torch.load(model_file)

    inp_len = data_json["len_input"]
    hidd_len = data_json["hidden_layer_len"]
    out_len = data_json["len_output"]
    all_words_data = data_json["b_o_a_w"]
    data_tags = data_json["tag"]
    mod_st = data_json["state"]

    model = NeruralNet(inp_len, hidd_len, out_len).to(device)
    model.load_state_dict(mod_st)
    #model.eval()



    assistance = "Assistance"
    #print("Let me know what you want, type quit to exit")

#while True:
    #user_input = input('You: ')
    #if user_input =="quit":
        #break

    # tokenize the input from the user
    sentance = tok_sentance(user_input)
    # creating the bag of words using the sentace
    test_data = b_o_w(sentance, all_words_data)
    # reshaping the X value
    test_data = test_data.reshape(1, test_data.shape[0])
    test_data = torch.from_numpy(test_data)


    predicton = model(test_data)
    _, predicted = torch.max(predicton,dim=1)
    # predict the tag using torch.max function
    prediction_tag = tags[predicted.item()]
    print(prediction_tag)

    # probabilty = create a softmax layer to provide the probability or percentage
    probability = torch.softmax(predicton, dim=1)
    # predict proba
    proba = probability[0][predicted.item()]

    # if proba  > 60% perform the following task
    if proba.item() >= 0.60:
        for data in intent_file['data']:
            if prediction_tag == data["tagged_data"]:
                return random.choice(data['bot_response'])
                #print("{}: {}".format(assistance,random.choice(data['bot_response'])))

    else:
        return "{}: Can you repeat that again...".format(assistance)