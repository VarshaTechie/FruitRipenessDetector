import os
import argparse
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define class names corresponding to label indices
CLASS_NAMES = ['Ripe', 'Rotten', 'Unripe']
IMAGE_SIZE = (224, 224)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Fruit Ripeness Detection Model using MobileNetV2 Transfer Learning")
    parser.add_argument('--dataset_dir', type=str, default='dataset/train', help='Path to dataset train directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the classification head')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--model_path', type=str, default='fruit_ripeness_model.keras', help='Path to save the trained model')
    parser.add_argument('--metadata_path', type=str, default='model_metadata.json', help='Path to save the class index mapping metadata')
    parser.add_argument('--plot_path', type=str, default='training_history.png', help='Path to save the training history plot')
    parser.add_argument('--fine_tune', action='store_true', help='If set, fine-tunes top layers of MobileNetV2')
    parser.add_argument('--fine_tune_epochs', type=int, default=5, help='Number of epochs for fine-tuning')
    return parser.parse_args()

def load_dataset_metadata(dataset_dir):
    """
    Scans the dataset directory and maps each folder starting with Ripe, Rotten, Unripe
    to class labels 0, 1, 2 without moving files.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    filepaths = []
    labels = []
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
        
    print(f"Scanning dataset directory: {dataset_dir}")
    folder_counts = {}
    
    for folder in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Class mapping based on folder prefix
        if folder.startswith('Ripe'):
            label_idx = 0
        elif folder.startswith('Rotten'):
            label_idx = 1
        elif folder.startswith('Unripe'):
            label_idx = 2
        else:
            print(f"Skipping folder: {folder} (does not start with Ripe, Rotten, or Unripe)")
            continue
            
        folder_files = 0
        for file in os.listdir(folder_path):
            if file.lower().endswith(valid_extensions):
                filepaths.append(os.path.abspath(os.path.join(folder_path, file)))
                labels.append(label_idx)
                folder_files += 1
                
        if folder_files > 0:
            folder_counts[folder] = folder_files

    print(f"\nFound {len(filepaths)} total images across {len(folder_counts)} folders.")
    print("Class mappings and distribution:")
    class_totals = [0, 0, 0]
    for idx, name in enumerate(CLASS_NAMES):
        class_totals[idx] = labels.count(idx)
        print(f"  Class {idx} ({name}): {class_totals[idx]} images")
        
    return filepaths, labels

def preprocess_image(filepath, label):
    """
    Loads, decodes, resizes, and preprocesses an image for MobileNetV2.
    """
    img_raw = tf.io.read_file(filepath)
    # decode_image handles JPEG, PNG, BMP, etc.
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label

def build_dataset(filepaths, labels, batch_size, is_training=True):
    """
    Creates an optimized tf.data.Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    # Map preprocessing function
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1024)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def build_model(num_classes=3):
    """
    Builds the transfer learning model with MobileNetV2 base.
    """
    print("\nBuilding model with MobileNetV2 base...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Build classification head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def save_history_plot(history, plot_path, has_fine_tuned=False):
    """
    Generates and saves the loss and accuracy plot.
    """
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved training history plot to {plot_path}")
    plt.close()

def main():
    args = parse_args()
    
    # 1. Load dataset metadata
    try:
        filepaths, labels = load_dataset_metadata(args.dataset_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    if len(filepaths) == 0:
        print("No valid images found. Please verify the dataset directory structure.")
        return
        
    # 2. Stratified train-val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        filepaths, labels,
        test_size=args.val_split,
        stratify=labels,
        random_state=42
    )
    print(f"Split into {len(train_paths)} training images and {len(val_paths)} validation images.")
    
    # 3. Create datasets
    train_dataset = build_dataset(train_paths, train_labels, args.batch_size, is_training=True)
    val_dataset = build_dataset(val_paths, val_labels, args.batch_size, is_training=False)
    
    # 4. Build and compile the transfer learning model
    model, base_model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.summary()
    
    # 5. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 6. Train the classification head
    print(f"\nPhase 1: Training the classification head for {args.epochs} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # 7. Optional Fine-tuning phase
    if args.fine_tune:
        print("\nPhase 2: Fine-tuning. Unfreezing MobileNetV2 base layers...")
        # Unfreeze base model
        base_model.trainable = True
        
        # We only want to fine-tune top layers, freeze lower ones
        # MobileNetV2 has 154 layers in total. Let's freeze up to layer 100
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Recompile model with lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate * 0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        model.summary()
        
        # We can adjust early stopping patience for fine tuning
        ft_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=args.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"Fine-tuning for {args.fine_tune_epochs} epochs...")
        ft_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=history.epoch[-1] + 1,
            callbacks=ft_callbacks
        )
        
        # Merge histories
        for key in history.history:
            history.history[key].extend(ft_history.history[key])
            
    # 8. Save class mappings and metadata
    metadata = {
        'class_names': CLASS_NAMES,
        'image_size': IMAGE_SIZE,
        'class_mapping': {idx: name for idx, name in enumerate(CLASS_NAMES)}
    }
    with open(args.metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to {args.metadata_path}")
    
    # 9. Plot and save training curves
    save_history_plot(history, args.plot_path, has_fine_tuned=args.fine_tune)
    
    # 10. Display final evaluation
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Model saved to: {args.model_path}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
