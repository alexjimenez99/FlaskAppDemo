from flask import Flask, render_template, request, jsonify
from LangModel import LangModel
# from chat import get_response

app = Flask(__name__)
# @app.get('/')
@app.route('/')
def index_get():
    return render_template('base.html')

# @app.post('/predict')
@app.route('/predict', methods=['POST'])
@app.route('/chat_message', methods=['POST'])
def predict():
    data     = request.get_json()
    message  = data['message']
    response = LangModel().get_model(message)
    # text = request.get_json().get('message')
    # response = get_response(text)
    message  = {'answer': response}
    return message

if __name__ == '__main__':
    app.run(debug = True)
    