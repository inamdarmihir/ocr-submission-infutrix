#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate TrOCR model performance.
This script:
1. Creates a 'model_files' directory
2. Organizes all model files into this directory
3. Loads the model and evaluates its performance on test data
"""

import os
import shutil
import glob
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset

# Create a directory for model files
def organize_model_files():
    model_dir = Path("model_files")
    model_dir.mkdir(exist_ok=True)
    print(f"Created/found model directory: {model_dir}")
    
    # Look for model files in current directory
    model_patterns = [
        "*.pt", "*.pth", "*.bin", "*.safetensors",
        "checkpoint-*", "fine_tuned_*"
    ]
    
    found_files = []
    for pattern in model_patterns:
        found_files.extend(glob.glob(pattern))
    
    # Move model directories
    for item in found_files:
        if os.path.isdir(item):
            # For directories, copy the entire directory
            target_dir = model_dir / os.path.basename(item)
            if not target_dir.exists():
                shutil.copytree(item, target_dir)
                print(f"Copied model directory: {item} -> {target_dir}")
            else:
                print(f"Directory already exists: {target_dir}")
        elif os.path.isfile(item):
            # For files, copy to the model directory
            target_file = model_dir / os.path.basename(item)
            if not target_file.exists():
                shutil.copy(item, target_file)
                print(f"Copied model file: {item} -> {target_file}")
            else:
                print(f"File already exists: {target_file}")
    
    return model_dir

# Load model and processor
def load_model(model_path=None):
    try:
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            processor = TrOCRProcessor.from_pretrained(model_path)
            model = VisionEncoderDecoderModel.from_pretrained(model_path)
        else:
            print("Loading base Microsoft TrOCR model")
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model loaded successfully on {device}")
        
        return model, processor, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# Evaluate model on a sample from the test set
def evaluate_model(model, processor, device):
    print("Loading test dataset...")
    try:
        # Load a few test samples from Teklia/IAM-line
        dataset = load_dataset("Teklia/IAM-line", split="test[:50]")
        
        # Define metrics
        total_cer = 0
        total_wer = 0
        
        # Levenshtein distance for CER/WER calculation
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # Process each sample and calculate metrics
        print("\nEvaluating model on test samples...")
        num_samples = min(10, len(dataset))  # Evaluate on 10 samples
        results = []
        
        for i in range(num_samples):
            sample = dataset[i]
            image = sample["image"]
            reference = sample["text"]
            
            # Preprocess image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            
            # Generate prediction
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values, max_length=64)
            prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Calculate metrics
            char_error = levenshtein_distance(prediction, reference) / max(len(reference), 1)
            word_error = levenshtein_distance(prediction.split(), reference.split()) / max(len(reference.split()), 1)
            
            total_cer += char_error
            total_wer += word_error
            
            results.append((image, reference, prediction, char_error, word_error))
            
            print(f"Sample {i+1}:")
            print(f"  Reference: {reference}")
            print(f"  Prediction: {prediction}")
            print(f"  CER: {char_error:.4f}, WER: {word_error:.4f}")
            print("-" * 50)
        
        # Calculate average metrics
        avg_cer = total_cer / num_samples
        avg_wer = total_wer / num_samples
        
        print(f"\nAverage CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        
        # Visualize some results
        visualize_results(results[:5])  # Show first 5 results
        
        return avg_cer, avg_wer
    
    except Exception as e:
        print(f"Error evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return 1.0, 1.0

# Visualize some results
def visualize_results(results):
    try:
        plt.figure(figsize=(15, 4 * len(results)))
        
        for i, (image, reference, prediction, cer, wer) in enumerate(results):
            plt.subplot(len(results), 1, i+1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Reference: {reference}\nPrediction: {prediction}\nCER: {cer:.4f}, WER: {wer:.4f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("evaluation_results.png")
        print(f"Visualization saved to: evaluation_results.png")
    except Exception as e:
        print(f"Error visualizing results: {e}")

def main():
    # Organize model files
    model_dir = organize_model_files()
    
    # Find the most recent model checkpoint
    checkpoints = list(model_dir.glob("checkpoint-step-*")) + list(model_dir.glob("fine_tuned_*"))
    if checkpoints:
        # Sort by modification time (most recent first)
        latest_checkpoint = sorted(checkpoints, key=lambda p: os.path.getmtime(p), reverse=True)[0]
        print(f"Found latest checkpoint: {latest_checkpoint}")
        model_path = latest_checkpoint
    else:
        model_path = None
        print("No checkpoints found, will use base model")
    
    # Load model
    model, processor, device = load_model(model_path)
    
    if model and processor:
        # Evaluate model
        cer, wer = evaluate_model(model, processor, device)
        
        # Assessment
        if cer <= 0.07 and wer <= 0.15:
            print("\n✅ Model performance meets the target metrics!")
        else:
            print("\n⚠️ Model performance doesn't meet the target metrics yet.")
            print("   Target: CER ≤ 7%, WER ≤ 15%")
        
        print("\nEvaluation complete!")
    else:
        print("Failed to load model for evaluation.")

if __name__ == "__main__":
    main() 