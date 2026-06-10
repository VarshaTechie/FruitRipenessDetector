# Fruit Ripeness Detection 🍎
ripe-check.streamlit.app

A deep learning powered web application built with Streamlit and TensorFlow that instantly detects whether a fruit is **Ripe**, **Rotten**, or **Unripe**.

## Features
- **Deep Learning Model:** Utilizes MobileNetV2 transfer learning for high accuracy and fast predictions.
- **Modern UI/UX:** Clean, responsive interface with beautiful micro-interactions, drag-and-drop uploads, and in-place image previews.
- **Instant Analytics:** Detailed class probability breakdown and confidence scores directly below the image.
- **Smart Recommendations:** Gives you actionable advice based on the state of the fruit (e.g., "Ready to eat", "Discard fruit").
- **Sample Testing:** One-click buttons to instantly load sample images and see how the app works without needing your own photos.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VarshaTechie/FruitRipenessDetector.git
   cd FruitRipenessDetection
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # source venv/bin/activate    # On macOS/Linux
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install tensorflow
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   The app will automatically open in your default web browser at `http://localhost:8501`.

## File Structure
- `app.py`: The main Streamlit web application.
- `train.py`: Script to train the MobileNetV2 model on your dataset.
- `analyze.py`: Advanced model evaluation script to generate Grad-CAM heatmaps and confusion matrices.
- `fruit_ripeness_model.keras`: The compiled and trained TensorFlow model.
- `model_metadata.json`: Contains the class names mapping.

## Technologies Used
- Python
- TensorFlow / Keras (MobileNetV2)
- Streamlit
- Pillow
- NumPy

## License
MIT License
