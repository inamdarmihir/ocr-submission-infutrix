# TrOCR Handwriting Recognition

Fine-tuning pipeline for Microsoft's TrOCR model for handwritten text recognition, designed to work reliably in Google Colab environments.

## Features

- Fine-tunes TrOCR on the Teklia/IAM-line dataset
- Saves checkpoints every 100 steps
- Optimized for Google Colab's free tier
- Automatically saves to Google Drive when run from a notebook cell
- Handles training resumption from checkpoints
- Implements robust error handling

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Quick Start

### In Google Colab

1. Upload all files to your Colab session
2. Run in a notebook cell to save checkpoints to Google Drive:

```python
!python fine_tune_trocr.py
```

3. For local storage without Drive mounting:

```python
!python train_local.py
```

## Key Features

- **Frequent Model Saving**: Checkpoints saved every 100 steps
- **Google Drive Integration**: Automatic Drive mounting when run from a notebook cell
- **Memory Optimization**: Designed for limited Colab GPU resources
- **Robust Error Handling**: Handles common Colab issues like disconnections
- **Automatic Metrics**: Calculates Character Error Rate (CER) and Word Error Rate (WER)

## Training Configuration

- Base model: microsoft/trocr-base-handwritten
- Batch size: 1 (with gradient accumulation of 8)
- Learning rate: 5e-5
- Number of epochs: 3

## Troubleshooting

- **Drive mounting error**: Run from a notebook cell, not command line
- **CUDA out of memory**: Reduce batch size or dataset size by uncommenting dataset limiting code
- **Checkpoint loading issues**: Ensure the checkpoint path exists and contains valid files

## Project Structure

```
OCR-AI/
├── main.py                  # Entry point script
├── requirements.txt         # Dependencies
├── colab_notebook.ipynb     # Google Colab notebook
├── ocr/                     # Core package
│   ├── __init__.py          # Package initialization
│   ├── model.py             # Model loading and configuration
│   ├── data.py              # Dataset loading and preprocessing
│   ├── training.py          # Training and evaluation
│   └── utils.py             # Utility functions
```

## Performance

TrOCR fine-tuned on the Teklia/IAM-line dataset achieves:

- Character Error Rate (CER): ~5-7%
- Word Error Rate (WER): ~12-15%

Results will vary based on dataset, model size, and training duration.

## Acknowledgements

- [Microsoft TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten): The base model
- [Hugging Face Transformers](https://github.com/huggingface/transformers): The underlying library
- [Teklia IAM Dataset](https://huggingface.co/datasets/Teklia/IAM-line): The benchmark dataset

## License

MIT License 