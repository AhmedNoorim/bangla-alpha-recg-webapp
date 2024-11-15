from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageFilter
import io
import os

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "./font/Li Shadhinata 2.0 Unicode.ttf"
prop_font = fm.FontProperties(fname=font_path)
print(prop_font)

# Use a non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
model = load_model("./model/CNN1_e50_mypc.h5")  # Load your trained model

# Ensure a directory exists for saving plots
if not os.path.exists('static'):
    os.makedirs('static')

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img_resized = img.resize((60, 60), resample=Image.Resampling.LANCZOS)
    img_rgb = img_resized.convert("RGB")
    img_blurred = img_rgb.filter(ImageFilter.GaussianBlur(radius=2))
    img_weighted = Image.blend(img_rgb, img_blurred, alpha=0.5)
    sharpen_filter = ImageFilter.UnsharpMask(radius=20, percent=150)
    img_sharpened = img_weighted.filter(sharpen_filter)
    threshold_value = 128
    img_binary = img_sharpened.point(lambda p: 255 if p > threshold_value else 0)
    img_array = np.array(img_binary)
    img_array = img_array.astype('float32') / 255.0
    return img_array

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img_bytes = file.read()
    processed_image = preprocess_image(img_bytes)

    predictions = model.predict(np.expand_dims(processed_image, axis=0))
    
    predicted_class_index = np.argmax(predictions[0])
    
    class_indices = {
        0: 'ং', 1: 'ঃ', 2: 'অ', 3: 'আ', 4: 'ই',
        5: 'ঈ', 6: 'উ', 7: 'ঊ', 8: 'ঋ', 9: 'এ',
        10: 'ঐ', 11: 'ও', 12: 'ঔ', 13: 'ক', 14: 'খ',
        15: 'গ', 16: 'ঘ', 17: 'ঙ', 18: 'চ', 19: 'ছ',
        20: 'জ', 21: 'ঝ', 22: 'ঞ', 23: 'ট', 24: 'ঠ',
        25: 'ড', 26: 'ড়', 27: 'ঢ', 28: 'ঢ়', 29: 'ণ',
        30: 'ত', 31: 'থ', 32: 'দ', 33: 'ধ', 34: 'ন',
        35: 'প', 36: 'ফ', 37: 'ব', 38: 'ভ', 39: 'ম',
        40: 'য', 41: 'য়', 42: 'র', 43: 'ল', 44: 'শ',
        45: 'ষ', 46: 'স', 47: 'হ', 48: 'ৎ', 49: '‍ঁ',
    }

    predicted_class_name = class_indices.get(predicted_class_index, "Give Bengali")

    # Prepare data for plotting
    probabilities = [prob *100 for prob in predictions[0]] # Convert to percentage
    labels = [class_indices[i] for i in range(len(probabilities))]

    try:
        # Create a bar graph and save it as an image
        plt.figure(figsize=(10,6))
        plt.bar(labels, probabilities, color='skyblue')
        plt.ylim(0, max(probabilities) + 10)
        
        
        # Check if font path exists before using it
        if os.path.exists(font_path):
            
            plt.xticks(fontproperties=prop_font)
            plt.yticks(fontproperties=prop_font)
            plt.title('Top Probabilities', fontproperties=prop_font)
            plt.ylabel('Probability (%)', fontproperties=prop_font)
            plt.xlabel('Class Labels', fontproperties=prop_font)
        else:
            plt.title('Top Probabilities')
            plt.ylabel('Probability (%)')
            plt.xlabel('Class Labels')

        # Save the figure
        plot_path = "static/prediction_plot.png"
        plt.savefig(plot_path)

        # Clear the figure after saving to avoid overlap in future plots
        plt.clf()

    except Exception as e:
        print(f"Error while creating plot: {e}")
        return jsonify({"error": "Failed to create plot"}),500

    print(f"Predicted Class Index: {predicted_class_index}")
    print(f"Predicted Class Name: {predicted_class_name}")

    return jsonify({"prediction": predicted_class_name, "plot_url": plot_path})

if __name__ == "__main__":
    app.run(debug=True)