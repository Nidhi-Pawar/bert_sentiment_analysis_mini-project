# Sentiment Analysis on SST-2 using BERT + LSTM
Fine-tuning bert-base-uncased with an LSTM classification head on the Stanford Sentiment Treebank v2 (SST-2) dataset.
---
### 📚 Project Overview
This project fine-tunes a pre-trained BERT model with an additional LSTM layer for the task of sentiment classification (positive/negative) using the SST-2 dataset.
It demonstrates:
- Efficient tokenization and dynamic padding
- Custom BERT+LSTM model creation
- Fine-tuning with CUDA and mixed-precision (FP16) training
- Regularization, learning rate scheduling, and early stopping to prevent overfitting
- Evaluation using Accuracy and F1-score metrics
---
### 🛠 Tech Stack
- HuggingFace Transformers (bert-base-uncased)
- PyTorch (with torch.cuda.amp for mixed-precision training)
- scikit-learn (for evaluation metrics)
- Matplotlib (for training/validation plots)
---

### 🧩 Project Structure
```
bert_sentiment_analysis-mini-project/
│
├──notebooks
│    ├── BERT_Sentiment_Analysis.ipynb
│    └── BERT_Sentiment_Analysis.pdf 
├── README.md                        
├── requirements.txt                 
└── assets
      ├── Training and Validation Loss.png
      ├── Training and Validation Accuracy.png
      └── Training and Validation F1 Score.png
```
---                 
### 🧪 How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```
2. Open the notebook:
```
jupyter notebook sentiment_analysis_bert.ipynb
```
---

### 🏆 Highlights
- Applied dynamic padding and truncation for efficient tokenization.
- Integrated early stopping and learning rate scheduling.
- Used automatic mixed precision (AMP) for faster training on GPU.
- Performed detailed error handling during training and evaluation.
---
### ⚡Future Work
- Experiment with different LSTM depths and hidden sizes.
- Try other pre-trained models (e.g., roberta-base).
- Hyperparameter tuning with libraries like Optuna.
---
### 📜 License
This project is open for learning and educational purposes.



