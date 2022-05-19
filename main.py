from crypt import methods
import os
import logging
from flask import Flask, request, render_template
import random
import json
from chatbot.chat import chatbot


app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/intent', methods=['POST'])
def intent():
    user_input = request.get_json()
    print(user_input['data'])
    res, tag = chatbot(user_input['data'])
    print(res)
    data = {"res": res, "tag": tag}
    return json.dumps(data)
    #return "user input: {} \n assistance: {}".format(user_input['data'], res)

@app.errorhandler(500)
def server_error(e):
    logging.exception('ERROR!')
    return """
 An error occurred: <pre>{}</pre>
 """.format(e), 500


if __name__ == '__main__':
 app.run(host='127.0.0.1', port=8080, debug=True)