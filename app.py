from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__, static_url_path='/static')

# ========== Cấu hình ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "odir5k_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

class_names = ["ageDegeneration", "cataract", "diabetes", "glaucoma", "hypertension", "myopia", "normal", "diabetic retinopathy", "retinitis pigmentosa", "disc edema", "pterygium", "retinal detachment"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========== Load model ==========
model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Model loaded successfully.")

    num_model_classes = model.layers[-1].output_shape[-1]
    if num_model_classes != len(class_names):
        raise ValueError(f"Số lớp không khớp! Model có {num_model_classes}, class_names có {len(class_names)}")

except Exception as e:
    print(f" Lỗi khi load model: {e}")


# ========== Routes ==========
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model chưa load thành công."

    if 'file' not in request.files:
        return "Không có file upload."

    file = request.files['file']
    if file.filename == '':
        return "Chưa chọn file."

    try:
        # Lưu file upload
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # Tiền xử lý ảnh
        img = Image.open(filepath).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Dự đoán
        predictions = model.predict(img_array)[0]  # lấy vector xác suất
        percentages = {cls: float(prob*100) for cls, prob in zip(class_names, predictions)}

        # Sắp xếp theo xác suất giảm dần
        sorted_probs = dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))

        # Trả về HTML
        img_url = f"/static/uploads/{unique_filename}"
        return render_template(
            "index.html",
            predictions=sorted_probs,
            img_path=img_url
        )

    except Exception as e:
        return f"Lỗi xử lý ảnh: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)