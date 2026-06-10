"""
Fruit Ripeness Detection - Model Analysis & Grad-CAM Visualization
Generates:
  1. Grad-CAM heatmaps for sample images
  2. Confusion matrix
  3. Misclassified image gallery
  4. Per-class precision, recall, F1-score
  5. Recommendations to reduce false rotten predictions
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

CLASS_NAMES = ['Ripe', 'Rotten', 'Unripe']
IMAGE_SIZE = (224, 224)
OUTPUT_DIR = 'analysis_output'


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Fruit Ripeness Model")
    parser.add_argument('--dataset_dir', type=str, default='dataset/train')
    parser.add_argument('--model_path', type=str, default='fruit_ripeness_model.keras')
    parser.add_argument('--metadata_path', type=str, default='model_metadata.json')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--max_gradcam', type=int, default=12, help='Max Grad-CAM samples to generate')
    parser.add_argument('--max_misclassified', type=int, default=20, help='Max misclassified images to show')
    return parser.parse_args()


def load_dataset_metadata(dataset_dir):
    """Scan dataset folders and map to class labels."""
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    filepaths, labels = [], []

    for folder in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder.startswith('Ripe'):
            label = 0
        elif folder.startswith('Rotten'):
            label = 1
        elif folder.startswith('Unripe'):
            label = 2
        else:
            continue

        for f in os.listdir(folder_path):
            if f.lower().endswith(valid_ext):
                filepaths.append(os.path.abspath(os.path.join(folder_path, f)))
                labels.append(label)

    return filepaths, labels


def load_and_preprocess(filepath):
    """Load a single image, resize, and preprocess for MobileNetV2."""
    img = Image.open(filepath).convert('RGB')
    img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    processed = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr, processed


# ─────────────────────────────────────────────
# 1. GRAD-CAM
# ─────────────────────────────────────────────

def make_gradcam_heatmap(model, img_array, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image.
    Works by finding the last Conv2D layer inside the MobileNetV2 base.
    """
    # Find the MobileNetV2 base model (functional sub-model)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find MobileNetV2 base model layer")

    # Find the last conv layer in the base model
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in base model")

    # Build a model that maps input -> [last_conv_output, predictions]
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[base_model.get_layer(last_conv_layer.name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = Image.fromarray(heatmap_resized).resize(
        (original_img.shape[1], original_img.shape[0]), Image.Resampling.BILINEAR
    )
    heatmap_resized = np.array(heatmap_resized)

    cmap = plt.cm.jet
    colored_heatmap = cmap(heatmap_resized / 255.0)[:, :, :3]
    colored_heatmap = np.uint8(255 * colored_heatmap)

    original_uint8 = np.uint8(original_img)
    superimposed = np.uint8(alpha * colored_heatmap + (1 - alpha) * original_uint8)
    return superimposed


def generate_gradcam_gallery(model, filepaths, labels, output_dir, max_samples=12):
    """Generate a Grad-CAM gallery for sample images from each class."""
    print("\n[1/5] Generating Grad-CAM heatmaps...")
    os.makedirs(output_dir, exist_ok=True)

    # Pick samples from each class
    samples_per_class = max(1, max_samples // 3)
    selected = []
    for cls_idx in range(3):
        cls_files = [(fp, lb) for fp, lb in zip(filepaths, labels) if lb == cls_idx]
        np.random.seed(42)
        indices = np.random.choice(len(cls_files), min(samples_per_class, len(cls_files)), replace=False)
        for i in indices:
            selected.append(cls_files[i])

    n = len(selected)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if rows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for idx, (filepath, true_label) in enumerate(selected):
        original, processed = load_and_preprocess(filepath)
        batch = np.expand_dims(processed, axis=0)
        preds = model.predict(batch, verbose=0)[0]
        pred_idx = np.argmax(preds)
        confidence = preds[pred_idx]

        heatmap = make_gradcam_heatmap(model, batch, pred_index=pred_idx)
        superimposed = overlay_heatmap(original, heatmap)

        ax = axes[idx]
        ax.imshow(superimposed)
        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_idx]
        color = 'green' if true_label == pred_idx else 'red'
        ax.set_title(f"True: {true_name}\nPred: {pred_name} ({confidence*100:.1f}%)",
                      fontsize=10, color=color, fontweight='bold')
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.suptitle("Grad-CAM Heatmaps — Model Attention Regions", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'gradcam_gallery.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path}")
    return path


# ─────────────────────────────────────────────
# 2. CONFUSION MATRIX
# ─────────────────────────────────────────────

def generate_confusion_matrix(y_true, y_pred, output_dir):
    """Plot and save the confusion matrix."""
    print("\n[2/5] Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    ax.set(xticks=[0, 1, 2], yticks=[0, 1, 2],
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           ylabel='True Label', xlabel='Predicted Label')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)

    # Annotate cells
    thresh = cm.max() / 2.
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cm[i, j]}',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Saved: {path}")
    return path


# ─────────────────────────────────────────────
# 3. MISCLASSIFIED IMAGE GALLERY
# ─────────────────────────────────────────────

def generate_misclassified_gallery(model, filepaths, y_true, y_pred, output_dir, max_images=20):
    """Show a gallery of misclassified images."""
    print("\n[3/5] Generating Misclassified Image Gallery...")
    misclassified = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            misclassified.append((filepaths[i], y_true[i], y_pred[i]))

    total_misclassified = len(misclassified)
    print(f"   Total misclassified: {total_misclassified} / {len(y_true)}")

    if total_misclassified == 0:
        print("   No misclassified images found!")
        return None

    # Pick up to max_images
    np.random.seed(42)
    if total_misclassified > max_images:
        indices = np.random.choice(total_misclassified, max_images, replace=False)
        misclassified = [misclassified[i] for i in indices]

    n = len(misclassified)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for idx, (fp, true_lb, pred_lb) in enumerate(misclassified):
        try:
            img = Image.open(fp).convert('RGB').resize((224, 224))
            ax = axes[idx]
            ax.imshow(np.array(img))
            ax.set_title(f"T:{CLASS_NAMES[true_lb]} → P:{CLASS_NAMES[pred_lb]}",
                          fontsize=8, color='red', fontweight='bold')
            ax.axis('off')
        except Exception:
            continue

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"Misclassified Images ({total_misclassified} total)", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'misclassified_gallery.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path}")
    return path


# ─────────────────────────────────────────────
# 4. CLASSIFICATION REPORT
# ─────────────────────────────────────────────

def generate_classification_report(y_true, y_pred, output_dir):
    """Generate and save per-class precision, recall, F1-score."""
    print("\n[4/5] Generating Classification Report...")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    # Save as text
    path = os.path.join(output_dir, 'classification_report.txt')
    with open(path, 'w') as f:
        f.write("Per-Class Precision, Recall, and F1-Score\n")
        f.write("=" * 55 + "\n\n")
        f.write(report)
    print(f"   Saved: {path}")

    # Also plot as a styled table
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    table_data = []
    for cls in CLASS_NAMES:
        d = report_dict[cls]
        table_data.append([cls, f"{d['precision']:.4f}", f"{d['recall']:.4f}",
                           f"{d['f1-score']:.4f}", f"{int(d['support'])}"])

    d = report_dict['weighted avg']
    table_data.append(['Weighted Avg', f"{d['precision']:.4f}", f"{d['recall']:.4f}",
                       f"{d['f1-score']:.4f}", f"{int(d['support'])}"])

    table = ax.table(cellText=table_data,
                     colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Style header
    for j in range(5):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        color = '#ecf0f1' if i % 2 == 0 else '#ffffff'
        for j in range(5):
            table[i, j].set_facecolor(color)

    plt.title("Classification Metrics", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    path_img = os.path.join(output_dir, 'classification_report.png')
    plt.savefig(path_img, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path_img}")
    return report


# ─────────────────────────────────────────────
# 5. RECOMMENDATIONS
# ─────────────────────────────────────────────

def generate_recommendations(y_true, y_pred, output_dir):
    """Analyze errors and produce recommendations."""
    print("\n[5/5] Generating Recommendations...")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    ripe_as_rotten = cm[0][1]   # Ripe misclassified as Rotten
    unripe_as_rotten = cm[2][1] # Unripe misclassified as Rotten
    rotten_as_ripe = cm[1][0]   # Rotten misclassified as Ripe

    total = len(y_true)
    total_misclassified = sum(1 for a, b in zip(y_true, y_pred) if a != b)
    accuracy = (total - total_misclassified) / total * 100

    recommendations = []
    recommendations.append("=" * 60)
    recommendations.append("RECOMMENDATIONS TO REDUCE FALSE ROTTEN PREDICTIONS")
    recommendations.append("=" * 60)
    recommendations.append("")
    recommendations.append(f"Overall Accuracy: {accuracy:.2f}%")
    recommendations.append(f"Total Misclassified: {total_misclassified} / {total}")
    recommendations.append(f"Ripe → Rotten errors: {ripe_as_rotten}")
    recommendations.append(f"Unripe → Rotten errors: {unripe_as_rotten}")
    recommendations.append(f"Rotten → Ripe errors: {rotten_as_ripe}")
    recommendations.append("")

    recommendations.append("─" * 60)
    recommendations.append("1. DATA AUGMENTATION")
    recommendations.append("─" * 60)
    recommendations.append("• Add random brightness, contrast, and saturation jitter")
    recommendations.append("  to training images. Ripe and Rotten fruits often differ")
    recommendations.append("  mainly in color/texture, and augmentation helps the model")
    recommendations.append("  learn more robust features instead of relying on lighting.")
    recommendations.append("• Use RandomFlip, RandomRotation, and RandomZoom layers.")
    recommendations.append("")

    recommendations.append("─" * 60)
    recommendations.append("2. FINE-TUNE THE BASE MODEL")
    recommendations.append("─" * 60)
    recommendations.append("• Run train.py with --fine_tune flag to unfreeze the top")
    recommendations.append("  layers of MobileNetV2 (layers 100+) and train with a")
    recommendations.append("  very small learning rate (1e-5).")
    recommendations.append("• This allows the model to adapt its feature extraction")
    recommendations.append("  to fruit-specific textures (bruising, discoloration).")
    recommendations.append("  Command: venv\\Scripts\\python.exe train.py --fine_tune")
    recommendations.append("")

    recommendations.append("─" * 60)
    recommendations.append("3. INCREASE TRAINING EPOCHS")
    recommendations.append("─" * 60)
    recommendations.append("• Train for 15-20 epochs with EarlyStopping (patience=5)")
    recommendations.append("  instead of the current 5 epochs to allow the model")
    recommendations.append("  more time to converge on subtle class boundaries.")
    recommendations.append("  Command: venv\\Scripts\\python.exe train.py --epochs 20")
    recommendations.append("")

    recommendations.append("─" * 60)
    recommendations.append("4. CLASS-SPECIFIC DATA BALANCING")
    recommendations.append("─" * 60)
    recommendations.append("• Use class weights during training to penalize")
    recommendations.append("  misclassifications of underrepresented classes more.")
    recommendations.append("• Oversample confusing pairs (Ripe/Rotten guavas)")
    recommendations.append("  or collect more diverse images for these categories.")
    recommendations.append("")

    recommendations.append("─" * 60)
    recommendations.append("5. REVIEW GRAD-CAM HEATMAPS")
    recommendations.append("─" * 60)
    recommendations.append("• Check gradcam_gallery.png to see WHERE the model looks.")
    recommendations.append("• If the model focuses on background instead of the fruit,")
    recommendations.append("  crop images tighter around the fruit before training.")
    recommendations.append("• If the model focuses on color patches, add color jitter")
    recommendations.append("  augmentation to force it to learn shape/texture features.")
    recommendations.append("")

    recommendations.append("─" * 60)
    recommendations.append("6. USE LABEL SMOOTHING")
    recommendations.append("─" * 60)
    recommendations.append("• Replace hard labels (0, 1, 2) with soft labels")
    recommendations.append("  (e.g., [0.9, 0.05, 0.05]) to reduce overconfidence")
    recommendations.append("  and help the model generalize better on edge cases")
    recommendations.append("  like partially ripe/rotten fruits.")
    recommendations.append("")

    text = "\n".join(recommendations)
    print(text)

    path = os.path.join(output_dir, 'recommendations.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(f"\n   Saved: {path}")
    return text


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    # Load metadata
    if os.path.exists(args.metadata_path):
        with open(args.metadata_path, 'r') as f:
            meta = json.load(f)
            global CLASS_NAMES
            CLASS_NAMES = meta.get('class_names', CLASS_NAMES)

    # Load dataset
    filepaths, labels = load_dataset_metadata(args.dataset_dir)
    print(f"Total images: {len(filepaths)}")

    # Stratified val split (same seed as training)
    _, val_paths, _, val_labels = train_test_split(
        filepaths, labels, test_size=args.val_split, stratify=labels, random_state=42
    )
    print(f"Validation set: {len(val_paths)} images")

    # Run predictions on validation set
    print("\nRunning predictions on validation set...")
    y_true = []
    y_pred = []
    for i, (fp, lb) in enumerate(zip(val_paths, val_labels)):
        try:
            _, processed = load_and_preprocess(fp)
            batch = np.expand_dims(processed, axis=0)
            preds = model.predict(batch, verbose=0)[0]
            pred_idx = np.argmax(preds)
            y_true.append(lb)
            y_pred.append(pred_idx)
        except Exception as e:
            print(f"   Skipping {fp}: {e}")

        if (i + 1) % 500 == 0:
            print(f"   Processed {i+1}/{len(val_paths)} images...")

    print(f"   Completed predictions for {len(y_true)} images.")

    # Generate all outputs
    generate_gradcam_gallery(model, val_paths, val_labels, args.output_dir, args.max_gradcam)
    generate_confusion_matrix(y_true, y_pred, args.output_dir)
    generate_misclassified_gallery(model, val_paths, y_true, y_pred, args.output_dir, args.max_misclassified)
    generate_classification_report(y_true, y_pred, args.output_dir)
    generate_recommendations(y_true, y_pred, args.output_dir)

    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print(f"All outputs saved to: {args.output_dir}/")
    print("  - gradcam_gallery.png")
    print("  - confusion_matrix.png")
    print("  - misclassified_gallery.png")
    print("  - classification_report.txt")
    print("  - classification_report.png")
    print("  - recommendations.txt")
    print("=" * 50)


if __name__ == "__main__":
    main()
