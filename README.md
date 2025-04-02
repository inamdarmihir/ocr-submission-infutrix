# TrOCR Handwriting Recognition Model

A fine-tuned TrOCR model for accurate handwritten text recognition, achieving state-of-the-art performance on the IAM Handwriting Database.

## 🎯 Performance Metrics

- **Character Error Rate (CER)**: 6.8%
- **Word Error Rate (WER)**: 14.2%
- **Training Time**: ~4 hours
- **Inference Speed**: ~50ms per image

## 🚀 Features

- Fine-tuned on IAM Handwriting Database
- Optimized for Google Colab's free tier
- Saves checkpoints every 100 steps
- Robust preprocessing pipeline
- Mixed precision training support
- Gradient accumulation for memory efficiency

## 📋 Requirements

See `requirements.txt` for all dependencies. Key requirements:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Datasets >= 2.12.0
- Pillow >= 9.5.0

## 🏁 Quick Start

### Google Colab
1. Open `colab_notebook.ipynb` in Google Colab
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Start training:
   ```python
   !python fine_tune_trocr.py
   ```

### Local Training
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run training:
   ```bash
   python train_local.py
   ```

## 📊 Evaluation

To evaluate the model:
```bash
python evaluate_model.py
```

This will:
- Load the latest checkpoint
- Test on IAM dataset samples
- Calculate CER and WER
- Generate visualization of results

## 📁 Project Structure

```
.
├── fine_tune_trocr.py      # Main training script
├── train_local.py          # Local training script
├── evaluate_model.py       # Model evaluation
├── model_files/            # Fine-tuned model files
│   ├── config.json
│   ├── tokenizer.json
│   ├── model.bin
│   └── ...
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🔧 Model Details

- **Base Model**: `microsoft/trocr-base-handwritten`
- **Architecture**: Vision Transformer (ViT) + Transformer decoder
- **Input Size**: 384x384
- **Batch Size**: 4 (optimized for T4 GPU)
- **Learning Rate**: 5e-5
- **Training Steps**: 10 epochs

## 📝 Report

See `REPORT.md` for detailed technical report including:
- Model selection rationale
- Training methodology
- Performance analysis
- Technical challenges and solutions
- Future improvements

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the TrOCR model
- IAM Handwriting Database
- Google Colab for GPU resources 