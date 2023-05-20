from flask import Flask, request
import argparse

from predict import phobeart_predict

parser = argparse.ArgumentParser(description='Choose option')
parser.add_argument('-p', '--port', type=int, default=8000)
parser.add_argument('-ht', '--host', type=str, default="0.0.0.0")
args = parser.parse_args()

app = Flask(__name__)

@app.post('/predict_label')
def check(): 
    txt = request.form.get('txt')
    label = phobeart_predict(txt)
    return {'label': label}

if __name__ == '__main__':
    port = args.port
    host = args.host
    app.run(host=host, port=port, debug=True)