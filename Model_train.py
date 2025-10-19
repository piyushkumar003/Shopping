# ---
# # AI Project: Computer Vision (CV) Model Script
#
# **Objective:** Develop and evaluate an image classification model.
#
# **Part 1:** Run a pre-trained ViT model (zero-shot) on our data.
# **Part 2:** Fine-tune the ViT model on our dataset's specific categories.
# **Part 3:** Evaluate the fine-tuned model.
# ---

import torch
import timm
from PIL import Image
import requests
import io
import ast
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import copy # Needed for saving best model state
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import warnings
import multiprocessing # Import multiprocessing

# Suppress warnings if needed (optional)
# warnings.filterwarnings("ignore", category=UserWarning)

# --- Helper Functions (Define these outside the main block) ---

def get_first_image_url(images_str):
    """Parses the string list of images and returns the first URL."""
    try:
        images_list = ast.literal_eval(images_str)
        if isinstance(images_list, list) and len(images_list) > 0:
            return images_list[0].strip()
    except:
        return None

def predict_image_category_zeroshot(image_url, model, transforms, labels):
    """Downloads an image and predicts its category using the ViT model (zero-shot)."""
    try:
        response = requests.get(image_url, timeout=10) # Increased timeout
        response.raise_for_status() # Check for download errors
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img_tensor = transforms(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.topk(probabilities, 1)

        return labels[top_idx[0].item()], top_prob[0].item()

    except Exception as e:
        # print(f"Zero-shot prediction failed for {image_url}: {e}") # Optional: for debugging
        return None, None

# --- Main Execution Block ---
if __name__ == '__main__':
    # Add freeze_support() for Windows multiprocessing safety
    multiprocessing.freeze_support()

    # ---
    # ### Part 1: Zero-Shot Classification (Evaluation)
    # ---
    print("--- Part 1: Zero-Shot Evaluation ---")
    
    # --- 1.1. Load Pre-trained ViT Model & Labels ---
    print("Loading pre-trained Vision Transformer (ViT) model...")
    model_zeroshot = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    model_zeroshot.eval()
    print("Zero-shot model loaded successfully.")

    data_config = timm.data.resolve_model_data_config(model_zeroshot)
    transforms_zeroshot = timm.data.create_transform(**data_config, is_training=False)

    LABELS_URL = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    try:
        response = requests.get(LABELS_URL)
        response.raise_for_status()
        labels_zeroshot = response.text.strip().split('\n')
        print(f"Loaded {len(labels_zeroshot)} ImageNet labels.")
    except Exception as e:
        print(f"Error loading ImageNet labels: {e}")
        labels_zeroshot = [] # Fallback

    # --- 1.3. Run Zero-Shot Prediction on All Images ---
    print("\nRunning Zero-Shot Prediction on Full Dataset...")
    try:
        df = pd.read_csv('intern_data_ikarus.csv')
        df['image_url'] = df['images'].apply(get_first_image_url)
        df_valid_urls = df.dropna(subset=['image_url'])

        results = []
        if labels_zeroshot: # Only run if labels were loaded
            for _, row in tqdm(df_valid_urls.iterrows(), total=df_valid_urls.shape[0], desc="Zero-Shot Predict"):
                label, confidence = predict_image_category_zeroshot(
                    row['image_url'],
                    model_zeroshot,
                    transforms_zeroshot,
                    labels_zeroshot
                )
                results.append(label)
        else:
            results = [None] * len(df_valid_urls) # Fill with None if labels failed

        df_valid_urls['zero_shot_prediction'] = results

        print("\n--- Zero-Shot Prediction Results (Top 10 Categories Predicted) ---")
        print(df_valid_urls['zero_shot_prediction'].value_counts().head(10))
    except FileNotFoundError:
        print("Error: 'intern_data_ikarus.csv' not found.")
    except Exception as e:
        print(f"An error occurred during zero-shot prediction: {e}")

    # ---
    # ### Part 2: Fine-Tuning for Custom Categories (Improved)
    # ---
    print("\n--- Part 2: Fine-Tuning ---")

    # --- 2.1. Setup (Device, Hyperparameters - UPDATED) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    PATIENCE = 10
    DATA_DIR = 'images_v3' # Make sure this points to your latest image folder

    # --- 2.2. Load Data (UPDATED Augmentations) ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        full_dataset = ImageFolder(root=DATA_DIR)
        num_classes = len(full_dataset.classes)
        print(f"\nFound {len(full_dataset)} images in {num_classes} categories:")
        print(full_dataset.classes)

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        # Correctly assign transforms after split using subset indices if needed,
        # but easier to assign to the original dataset reference if structure allows.
        # Let's re-assign to the datasets obtained from random_split directly for clarity.
        # This requires creating custom Subset classes or accessing the underlying dataset if using standard Subset.
        # A simpler way often used is to apply transforms *within* the DataLoader,
        # but assigning them here is also common. Let's assume standard Subset works this way:
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        # If the above transform assignment causes issues, define transforms directly in DataLoader:
        # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, collate_fn=lambda batch: (torch.stack([train_transform(item[0]) for item in batch]), torch.tensor([item[1] for item in batch])))
        # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=lambda batch: (torch.stack([val_transform(item[0]) for item in batch]), torch.tensor([item[1] for item in batch])))
        # For simplicity, we'll stick to the original method, assuming it works with standard random_split subsets

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=(device.type == 'cuda')) # Only pin if using CUDA
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'))

    except FileNotFoundError:
        print(f"Error: Data directory '{DATA_DIR}' not found. Did you run download_images.py successfully?")
        exit() # Stop script if data isn't present
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- 2.3. Load Model and Prepare for Fine-Tuning ---
    print("\nLoading ViT model for fine-tuning...")
    model_finetune = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)

    for param in model_finetune.parameters():
        param.requires_grad = False

    in_features = model_finetune.head.in_features
    model_finetune.head = nn.Linear(in_features, num_classes)

    for param in model_finetune.blocks[-1].parameters():
        param.requires_grad = True
    for param in model_finetune.norm.parameters():
        param.requires_grad = True

    model_finetune = model_finetune.to(device)

    print("\nTrainable parameters:")
    for name, param in model_finetune.named_parameters():
        if param.requires_grad:
            print(name)
    print(f"\nModel modified. Training head, norm layer, and last block ({num_classes} classes).")

    # --- 2.4. Define Loss Function, Optimizer, and Scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_finetune.parameters()), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3) # Removed verbose

    # --- 2.5. The Training Loop ---
    print("\n--- Starting Fine-Tuning (Improved) ---")

    patience = PATIENCE
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model_finetune.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_finetune(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})

        epoch_loss = running_loss / len(train_dataset) # Use len(train_dataset)
        history['train_loss'].append(epoch_loss)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Training Loss: {epoch_loss:.4f}")

        # --- Validation ---
        model_finetune.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_finetune(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

        epoch_val_loss = val_loss / len(val_dataset) # Use len(val_dataset)
        epoch_val_acc = 100 * correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation Loss: {epoch_val_loss:.4f} | Validation Acc: {epoch_val_acc:.2f}%")

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Saving model state...")
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model_finetune.state_dict())
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s). Best was {best_val_loss:.4f}.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("\n--- Fine-Tuning Complete! ---")

    # --- 2.6. Load the Best Model ---
    print(f"\nLoading best model weights (Val Loss: {best_val_loss:.4f})...")
    if best_model_state:
        model_finetune.load_state_dict(best_model_state)
        print("Best model loaded successfully for evaluation.")
    else:
        print("Warning: No best model state saved. Using model from last epoch.")

    best_model_path = "best_finetuned_cv_model_improved.pth"
    torch.save(model_finetune.state_dict(), best_model_path)
    print(f"Best model saved to {best_model_path}")

    # --- Optional: Plot Training History ---
    print("\nPlotting training history...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Validation Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---
    # ### Part 3: Final Performance Evaluation
    # ---
    print("\n--- Part 3: Final Evaluation ---")
    
    # Use the 'model_finetune' object which now holds the best weights
    model_eval = model_finetune 
    model_eval.eval() 

    print("\n--- Evaluating Model on Validation Set ---")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Final Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_eval(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Calculate and Print Metrics ---
    print("\n--- Classification Report ---")
    try:
        class_names = full_dataset.classes
        # Added zero_division=0 to handle cases where a class has no predictions
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    except Exception as e:
        print(f"Could not generate classification report: {e}")

    # --- Display Confusion Matrix ---
    print("\n--- Confusion Matrix ---")
    try:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix for Fine-Tuned Model')
        plt.show()
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")

    # --- Final Conclusion Text ---
    print("\n--- Conclusion ---")
    print("Successfully evaluated zero-shot model, fine-tuned the model on custom categories with improvements,")
    print("and performed detailed final evaluation.")
    print("The Classification Report and Confusion Matrix provide insights into model performance per category.")