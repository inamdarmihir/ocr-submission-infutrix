#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune TrOCR for handwritten text recognition
================================================
This script fine-tunes a TrOCR model on the Teklia/IAM-line dataset
for handwritten text recognition. It's optimized for use in Google Colab.

Author: AI Assistant
Date: 2023
"""

import os
import torch
import numpy as np
import gc
import time
import warnings
import sys
from datetime import datetime
from pathlib import Path
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    default_data_collator,
    TrainerCallback
)
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm

# Check if running in notebook (Jupyter/Colab) or standalone script
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal IPython
            return False
        else:
            return True  # Other type (?)
    except NameError:
        return False  # Standard Python interpreter

# Set up checkpoint resuming based on environment
resume_checkpoint = None

# Only parse command line arguments if running as standalone script, not in notebook
if not is_notebook():
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune TrOCR for handwritten text recognition')
    parser.add_argument('--resume_checkpoint', type=str, default=None, 
                        help='Path to checkpoint directory to resume training from, or "None" to start fresh')
    args = parser.parse_args()
    # Convert string 'None' to actual None
    if args.resume_checkpoint is None or args.resume_checkpoint.lower() == 'none':
        resume_checkpoint = None
    else:
        resume_checkpoint = args.resume_checkpoint
else:
    # For notebook environment, define a function to set the checkpoint
    def set_checkpoint_path(path):
        global resume_checkpoint
        if path is None or str(path).lower() == 'none':
            resume_checkpoint = None
            print("Set to start fresh training (no checkpoint)")
        else:
            resume_checkpoint = path
            print(f"Set checkpoint path to: {path}")

# If checkpoint was provided through command line
if resume_checkpoint:
    print(f"Will resume training from checkpoint: {resume_checkpoint}")

# Ignore non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Function to clear GPU memory
def free_gpu_memory():
    """Clear GPU memory cache and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared. Current allocation: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Check if running in Colab
IS_COLAB = False
IS_NOTEBOOK = is_notebook()

# Check for forced local storage mode
FORCE_LOCAL_STORAGE = os.environ.get("FORCE_LOCAL_STORAGE", "0") == "1"
if FORCE_LOCAL_STORAGE:
    print("Forced local storage mode enabled (environment variable)")

# Better detection of Colab environment and notebook execution context
try:
    from google.colab import drive
    IS_COLAB = True and not FORCE_LOCAL_STORAGE  # Disable Colab features if forced local
    print("Running in Google Colab environment")
    
    # Only try to mount Drive if we're in a notebook cell and not in forced local mode
    if IS_NOTEBOOK and not FORCE_LOCAL_STORAGE:
        try:
            print("Mounting Google Drive (interactive mode)...")
            drive.mount('/content/drive', force_remount=True)
            print("Google Drive mounted successfully!")
        except Exception as e:
            print(f"Failed to mount Google Drive: {e}")
            print("Will save output to local directory instead")
            IS_COLAB = False  # Revert to local saving behavior
    else:
        print("Running in Colab command line or forced local mode")
        print("To use Google Drive, run from a notebook cell without FORCE_LOCAL_STORAGE=1")
        print("Will save output to local directory instead")
        IS_COLAB = False  # Set to False to avoid Drive-dependent operations
except ImportError:
    print("Not running in Google Colab environment")

# Define device and enable reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) if torch.cuda.is_available() else None
np.random.seed(SEED)

# Create output directories with timestamp to avoid overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Save to Google Drive if running in Colab and Drive is available and not forced local
if IS_COLAB and IS_NOTEBOOK and os.path.exists('/content/drive') and not FORCE_LOCAL_STORAGE:
    drive_dir = Path("/content/drive/MyDrive/trocr_training")
    drive_dir.mkdir(parents=True, exist_ok=True)
    output_dir = drive_dir / f"fine_tuned_trocr_{timestamp}"
    logs_dir = drive_dir / "logs"
    print(f"Saving output to Google Drive: {output_dir}")
else:
    output_dir = Path(f"./fine_tuned_trocr_{timestamp}")
    logs_dir = Path("./logs")
    print(f"Saving output to local directory: {output_dir}")

output_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# Load the TrOCR processor and model with error handling
print("Loading TrOCR model...")
model = None
processor = None

try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Set proper model configuration for training
    # According to the official docs: https://huggingface.co/docs/transformers/en/model_doc/trocr
    model.config.decoder_start_token_id = processor.tokenizer.eos_token_id  # Use eos_token_id as recommended
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Also set in the decoder config to be safe
    if hasattr(model.config, 'decoder'):
        model.config.decoder.decoder_start_token_id = processor.tokenizer.eos_token_id
        model.config.decoder.pad_token_id = processor.tokenizer.pad_token_id
    
    # Verify the configuration
    print(f"Decoder start token ID set to: {model.config.decoder_start_token_id}")
    print(f"Pad token ID set to: {model.config.pad_token_id}")
    
    print("Base model loaded successfully")
except Exception as e:
    print(f"Error loading base model: {e}")
    try:
        print("Trying small model as fallback...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
        
        # Set proper model configuration for training (small model)
        model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        
        # Also set in the decoder config to be safe
        if hasattr(model.config, 'decoder'):
            model.config.decoder.decoder_start_token_id = processor.tokenizer.eos_token_id
            model.config.decoder.pad_token_id = processor.tokenizer.pad_token_id
        
        # Verify the configuration
        print(f"Decoder start token ID set to: {model.config.decoder_start_token_id}")
        print(f"Pad token ID set to: {model.config.pad_token_id}")
        
        print("Small model loaded successfully as fallback")
    except Exception as e2:
        print(f"Critical error loading models: {e2}")
        raise RuntimeError("Failed to load any TrOCR model. Check your internet connection.")

# Move model to device and clear memory
model.to(device)
free_gpu_memory()

# Load Teklia IAM dataset with retry logic
max_retries = 3
dataset = None

for attempt in range(max_retries):
    try:
        print(f"Loading dataset (attempt {attempt+1}/{max_retries})...")
        dataset = load_dataset("Teklia/IAM-line")
        print(f"Dataset loaded successfully with {len(dataset['train'])} training samples")
        print(f"Validation: {len(dataset['validation'])} samples, Test: {len(dataset['test'])} samples")
        break
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if attempt < max_retries - 1:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("Max retries reached. Could not load dataset.")
            raise

# Verify dataset structure
sample = dataset["train"][0]
print(f"Sample data structure: {', '.join([f'{k}: {type(v).__name__}' for k, v in sample.items()])}")

# For Teklia/IAM-line, the structure includes 'image' and 'text' fields
image_field = "image"
text_field = "text"
print(f"Using {image_field} for images and {text_field} for text")

# Option to reduce dataset size for Colab (uncomment if needed)
# max_train_samples = 2000
# max_val_samples = 300
# max_test_samples = 500
#
# if len(dataset["train"]) > max_train_samples:
#     dataset["train"] = dataset["train"].select(range(max_train_samples))
#     print(f"Limited training dataset to {max_train_samples} samples")
# if len(dataset["validation"]) > max_val_samples:
#     dataset["validation"] = dataset["validation"].select(range(max_val_samples))
#     print(f"Limited validation dataset to {max_val_samples} samples")
# if len(dataset["test"]) > max_test_samples:
#     dataset["test"] = dataset["test"].select(range(max_test_samples))
#     print(f"Limited test dataset to {max_test_samples} samples")

# Define image preprocessing function with error handling
def process_image(img):
    """Process a single image with error handling"""
    try:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        # Return a blank image as placeholder
        return Image.new('RGB', (100, 32), color='white')

# Preprocessing function for dataset
def preprocess_data(examples):
    """Process a batch of images and texts for TrOCR"""
    processed_images = []
    valid_indices = []
    
    # Process each image and track which ones are valid
    for i, img in enumerate(examples[image_field]):
        try:
            processed_img = process_image(img)
            processed_images.append(processed_img)
            valid_indices.append(i)
        except Exception as e:
            print(f"Error processing image {i}: {e}. Skipping.")
            # Skip this sample
    
    # If all images failed, add one placeholder to avoid errors
    if not processed_images:
        processed_images = [Image.new('RGB', (100, 32), color='white')]
        # Use an empty string if we have no valid texts
        valid_texts = [""]
        print("Warning: All images in batch failed processing!")
    else:
        # Get corresponding texts for valid images only
        valid_texts = [examples[text_field][i] for i in valid_indices]
    
    # Process images and texts with TrOCR processor
    try:
        # Process images
        pixel_values = processor(processed_images, return_tensors="pt", padding="max_length").pixel_values
        
        # Process texts to labels
        labels = processor.tokenizer(valid_texts, padding="max_length", return_tensors="pt").input_ids
        
        # Replace padding token id with -100 for loss calculation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "texts": valid_texts
        }
    except Exception as e:
        print(f"Error in processor: {e}")
        raise

# Process datasets
print("Preprocessing dataset...")
try:
    # Use a very small batch size for preprocessing to avoid memory issues
    batch_size = 2
    processed_datasets = dataset.map(
        preprocess_data,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing datasets"
    )
    print("Preprocessing complete")
except Exception as e:
    print(f"Error during preprocessing with batch size {batch_size}: {e}")
    try:
        print("Retrying with batch size 1...")
        processed_datasets = dataset.map(
            preprocess_data,
            batched=True,
            batch_size=1,
            remove_columns=dataset["train"].column_names,
            desc="Preprocessing with minimal batch"
        )
        print("Preprocessing complete with minimal batch size")
    except Exception as e2:
        print(f"Critical preprocessing error: {e2}")
        raise

free_gpu_memory()

# Load and prepare metrics
print("Setting up evaluation metrics...")
try:
    import evaluate
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        """Compute metrics with error handling for token overflow"""
        try:
            labels_ids = pred.label_ids
            pred_ids = pred.predictions
            
            # Replace -100 with pad token id for decoding
            labels_ids_copy = labels_ids.copy()
            labels_ids_copy[labels_ids_copy == -100] = processor.tokenizer.pad_token_id
            
            # Handle potentially problematic token IDs
            try:
                # Filter out extreme values from predictions
                safe_pred_ids = []
                for seq in pred_ids:
                    safe_seq = []
                    for token_id in seq:
                        if 0 <= token_id < processor.tokenizer.vocab_size:
                            safe_seq.append(token_id)
                        else:
                            safe_seq.append(processor.tokenizer.pad_token_id)
                    safe_pred_ids.append(safe_seq)
                
                # Decode predictions safely
                pred_str = []
                for ids in safe_pred_ids:
                    try:
                        decoded = processor.tokenizer.decode(ids, skip_special_tokens=True)
                        pred_str.append(decoded)
                    except Exception as e:
                        print(f"Error decoding prediction: {e}")
                        pred_str.append("")
                
                # Decode labels safely
                label_str = []
                for ids in labels_ids_copy:
                    try:
                        decoded = processor.tokenizer.decode(ids, skip_special_tokens=True)
                        label_str.append(decoded)
                    except Exception as e:
                        print(f"Error decoding label: {e}")
                        label_str.append("")
                
            except Exception as decode_error:
                print(f"Error during safe decoding: {decode_error}")
                # Fall back to custom handling
                pred_str = ["" for _ in range(len(pred_ids))]
                label_str = ["" for _ in range(len(labels_ids_copy))]
            
            # Calculate metrics
            # Calculate CER
            total_char_errors = 0
            total_chars = 0
            for p, l in zip(pred_str, label_str):
                total_char_errors += levenshtein_distance(p, l)
                total_chars += len(l)
            cer = total_char_errors / max(1, total_chars)
            
            # Calculate WER
            total_word_errors = 0
            total_words = 0
            for p, l in zip(pred_str, label_str):
                p_words = p.split()
                l_words = l.split()
                total_word_errors += levenshtein_distance(p_words, l_words)
                total_words += len(l_words)
            wer = total_word_errors / max(1, total_words)
            
            return {"cer": cer, "wer": wer}
        except Exception as e:
            print(f"Error in compute_metrics: {e}")
            import traceback
            traceback.print_exc()
            # Return default values
            return {"cer": 1.0, "wer": 1.0}

except Exception as e:
    print(f"Error setting up metrics: {e}")
    print("Using custom metrics implementation instead")
    
    # Simple Levenshtein distance implementation
    def levenshtein_distance(s1, s2):
        """Calculate Levenshtein distance between two strings"""
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
    
    def compute_metrics(pred):
        """Compute metrics using custom implementation"""
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        
        # Replace -100 with pad token id for decoding
        labels_ids_copy = labels_ids.copy()
        labels_ids_copy[labels_ids_copy == -100] = processor.tokenizer.pad_token_id
        
        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(labels_ids_copy, skip_special_tokens=True)
        
        # Calculate CER
        total_char_errors = 0
        total_chars = 0
        for p, l in zip(pred_str, label_str):
            total_char_errors += levenshtein_distance(p, l)
            total_chars += len(l)
        cer = total_char_errors / max(1, total_chars)
        
        # Calculate WER
        total_word_errors = 0
        total_words = 0
        for p, l in zip(pred_str, label_str):
            p_words = p.split()
            l_words = l.split()
            total_word_errors += levenshtein_distance(p_words, l_words)
            total_words += len(l_words)
        wer = total_word_errors / max(1, total_words)
        
        return {"cer": cer, "wer": wer}

# Configure training arguments for Colab
training_args = Seq2SeqTrainingArguments(
    output_dir=str(output_dir),
    eval_strategy="epoch",             # Still evaluate each epoch
    save_strategy="steps",             # Changed: Save by steps instead of epochs
    save_steps=100,                    # Save every 100 steps
    save_total_limit=5,                # Keep the 5 most recent checkpoints
    learning_rate=5e-5,                # Standard fine-tuning learning rate
    per_device_train_batch_size=1,     # Very small batch size for Colab
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    num_train_epochs=3,                # Short training for Colab
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),    # Use mixed precision if available
    logging_dir=str(logs_dir),
    logging_steps=50,
    gradient_accumulation_steps=8,     # Accumulate gradients to compensate for small batch
    load_best_model_at_end=True,       # Load the best model at the end of training
    metric_for_best_model="cer",       # Use CER as the metric for best model
    greater_is_better=False,           # Lower CER is better
    generation_max_length=64,
    generation_num_beams=1,            # Use greedy decoding to save memory
    report_to="none",                  # Disable wandb logging
)

# Note: If you want to use wandb logging, uncomment the following lines and set your API key:
# import os
# os.environ["WANDB_API_KEY"] = "your-api-key-here"
# training_args.report_to = ["wandb"]

# Custom callback to save model at specific steps
class SaveModelCallback(TrainerCallback):
    def __init__(self, save_every_n_steps, output_dir, processor):
        self.save_every_n_steps = save_every_n_steps
        self.output_dir = output_dir
        self.processor = processor
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Check if it's time to save
        if state.global_step % self.save_every_n_steps == 0 and state.global_step > 0:
            # Create a specific directory for this step
            step_dir = Path(self.output_dir) / f"checkpoint-step-{state.global_step}"
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model and processor
            if model is not None:
                model.save_pretrained(step_dir)
                self.processor.save_pretrained(step_dir)
                print(f"\n✓ Model saved at step {state.global_step} to {step_dir}\n")
                
                # Print the directory location for retrieval
                if IS_COLAB and "drive" in str(step_dir):
                    print(f"Google Drive path: {step_dir}")
            else:
                # If we don't have direct access to the model,
                # we'll use Hugging Face's built-in saving
                print(f"\nUsing built-in save for step {state.global_step}\n")
                control.should_save = True
                
        return control

# Initialize trainer with custom callbacks
print("Initializing trainer...")
# Create the custom save callback
save_callback = SaveModelCallback(
    save_every_n_steps=100,  # Save every 100 steps
    output_dir=output_dir,
    processor=processor
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer,
    callbacks=[save_callback]  # Add our custom callback
)
print("Trainer initialized successfully")

# Add a checkpoint note for Colab users
if IS_COLAB:
    print("\n----------------------------------------")
    print("IMPORTANT NOTE FOR COLAB USERS:")
    print("If Colab disconnects during training, you can reload this notebook")
    print("and resume training with:")
    print(f"  set_checkpoint_path('{output_dir}')")
    print("  # Or specify an exact checkpoint path:")
    print("  # set_checkpoint_path('fine_tuned_trocr_20250401_142752')")
    print("  # Then re-run the training cell")
    print("\nFor standalone script mode:")
    print("  python fine_tune_trocr.py --resume_checkpoint='fine_tuned_trocr_20250401_142752'")
    print("----------------------------------------\n")

# Check if a checkpoint path is provided
print(f"Checkpoint path for resume: {resume_checkpoint if resume_checkpoint else 'None (starting from scratch)'}")

# Fine-tune the model
print("Starting fine-tuning...")
start_time = time.time()

try:
    # Start or resume training - only pass resume_from_checkpoint if it's a valid path
    if resume_checkpoint and resume_checkpoint.lower() != 'none':
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        print("Starting fresh training (no checkpoint)")
        trainer.train()
    print("Fine-tuning complete!")
    
    # Check and report training time
    training_time = (time.time() - start_time) / 60
    print(f"Training took {training_time:.2f} minutes")
    
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        free_gpu_memory()
        print("\nCUDA out of memory error. Training failed due to GPU limitations.")
        print("Suggestions:")
        print("1. Reduce the training dataset size")
        print("2. Try a smaller model (trocr-small-handwritten)")
        print("3. Use Colab Pro+ for more GPU memory")
        # Re-raise with a clearer message
        raise RuntimeError("Training failed due to insufficient GPU memory") from e
    else:
        print(f"Error during training: {e}")
        raise
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving current state...")
    trainer.save_model(str(output_dir / "interrupted_model"))
    print(f"Interrupted model saved to {output_dir / 'interrupted_model'}")
    raise
except Exception as e:
    print(f"Unexpected error during training: {e}")
    raise

# Save the fine-tuned model
print("Saving model...")
try:
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Model and processor saved to {output_dir}")
except Exception as e:
    print(f"Error saving model: {e}")
    # Backup save method
    try:
        model_path = output_dir / "model_state_dict.pt"
        torch.save(model.state_dict(), str(model_path))
        print(f"Model state dict saved to {model_path}")
    except Exception as e2:
        print(f"Critical error saving model: {e2}")

free_gpu_memory()

# Evaluate on a subset of the test set to avoid memory issues
print("Evaluating on test set...")
try:
    # For Colab, use only a subset of test data
    test_subset_size = min(300, len(processed_datasets["test"]))
    test_subset = processed_datasets["test"].select(range(test_subset_size))
    
    test_results = trainer.evaluate(test_subset)
    print(f"Test results (on {test_subset_size} samples):")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")
    
    cer = test_results.get('eval_cer', float('nan'))
    wer = test_results.get('eval_wer', float('nan'))
    
    print(f"CER: {cer:.4f} (Target: ≤ 0.07)")
    print(f"WER: {wer:.4f} (Target: ≤ 0.15)")
    
    # Save results to file
    with open(output_dir / "evaluation_results.txt", "w") as f:
        f.write(f"Test results (on {test_subset_size} samples):\n")
        for metric, value in test_results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
except Exception as e:
    print(f"Error during evaluation: {e}")
    print("Skipping full evaluation due to errors")

# Function for inference
def recognize_handwriting(image_path, model=model, processor=processor, device=device):
    """
    Recognize text in a handwritten image
    
    Args:
        image_path: Path to the image file
        model: The TrOCR model
        processor: The TrOCR processor
        device: The device to run inference on
        
    Returns:
        Predicted text string
    """
    try:
        # Ensure path is a string
        image_path = str(image_path)
        
        # Check if file exists
        if not os.path.exists(image_path):
            return f"Error: File {image_path} does not exist"
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        try:
            # Try to display the image if in notebook environment
            from IPython.display import display
            display(image)
        except:
            pass
        
        # Clear memory before inference
        free_gpu_memory()
        
        # Prepare image for model
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        
        # Generate prediction conservatively
        generated_ids = model.generate(
            pixel_values, 
            max_length=64,
            num_beams=1,  # Greedy decoding for memory efficiency
            use_cache=False  # More memory efficient
        )
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return predicted_text
    except Exception as e:
        return f"Error: {str(e)}"

# Test on sample image if available
if IS_COLAB:
    try:
        # For Colab: Allow users to upload their own image
        from google.colab import files
        print("Upload a handwritten image for testing:")
        uploaded = files.upload()
        
        for filename in uploaded.keys():
            print(f"Processing {filename}...")
            predicted_text = recognize_handwriting(filename)
            print(f"Predicted Text: {predicted_text}")
    except Exception as e:
        print(f"Error with file upload: {e}")
else:
    # For non-Colab: Check for a sample image
    sample_image_path = "sample_image.png"
    if os.path.exists(sample_image_path):
        print(f"Testing on sample image: {sample_image_path}")
        predicted_text = recognize_handwriting(sample_image_path)
        print(f"Predicted Text: {predicted_text}")
    else:
        print("No sample image found for testing.")
        print("Please provide a sample image named 'sample_image.png'")

# Final cleanup
free_gpu_memory()
print(f"Script completed successfully! Model saved to {output_dir}") 