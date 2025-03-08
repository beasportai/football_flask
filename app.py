from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file, Response
from ultralytics import YOLO
import cv2
import numpy as np
import os
import main

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'output')
STUBS_FOLDER = 'stubs'

processed_player_metrics = {}


model = YOLO("final_best.pt")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    """Check if file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remfolder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):  
            os.remove(file_path)  

@app.route('/')
def index():
    remfolder(UPLOAD_FOLDER)
    remfolder(STUBS_FOLDER)
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    global processed_player_metrics
    if 'file' not in request.files:
        return redirect(url_for('index'))

    f = request.files['file']
    if f.filename == '':
        return redirect(url_for('index'))

    if not allowed_file(f.filename):  
        return redirect(url_for('index'))

    filepath = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(filepath)
    processed_player_metrics = main.main(filepath)
    processed_player_metrics = {int(k): v for k, v in processed_player_metrics.items()}
    
    return render_template('output.html')

@app.route('/get_player_ids', methods=['GET'])
def get_player_ids():
    return jsonify(list(processed_player_metrics.keys()))  # Return list of player IDs

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    player_id = request.args.get('player_id', type=int)
    metrics = processed_player_metrics.get(player_id, {})
    return jsonify(metrics)
if __name__ == '__main__':
    # port = int(os.environ.get("PORT", 5000))  # Render uses PORT from environment variables
    # app.run(host="0.0.0.0", port=port, debug=True)
    app.run()
