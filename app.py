from flask import Flask, request, jsonify, send_from_directory, render_template
import cv2
import numpy as np
import os
from ultralytics import YOLO

app = Flask(__name__)

# Загружаем модель YOLOv8 (наименьшая и быстрая версия)
model = YOLO('yolov8n.pt')

# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

# Обработка изображения
@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Детекция объектов
    results = model(img)
    output_img = results[0].plot()  # Отрисовка bbox на изображении

    # Сохранение результата
    result_path = 'static/result.jpg'
    cv2.imwrite(result_path, output_img)

    # Возврат количества найденных объектов
    return jsonify(count=len(results[0].boxes))

# Статические файлы (для результата)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
