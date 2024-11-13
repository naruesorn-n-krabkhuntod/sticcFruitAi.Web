from flask import Flask, jsonify, request
import os
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask API!"

@app.route('/orange', methods=['GET'])
def get_tasks():
    os.system("C:/ProgramData/anaconda3/envs/conda-env/python.exe detect.py")
    return jsonify({'tasks': "success"})

if __name__ == '__main__':
    app.run(debug=True)