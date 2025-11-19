# Sentiment Analysis (Amazon → Yelp Evaluation)

Fine-grained 3-class sentiment analysis using a BERT-base model trained on a balanced Amazon reviews dataset, with cross-domain evaluation on Yelp reviews.

## Overview

- Task: Classify text into three sentiment classes — Negative, Neutral, Positive.  
- Training data: 240,000 balanced Amazon review samples (80k per class).  
- Cross-domain test: Yelp Academic Dataset (sampled reviews).  
- Base model: `bert-base-uncased` fine-tuned with `AutoModelForSequenceClassification` (num_labels=3).

## Datasets

- Amazon US Customer Reviews (Kaggle): https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset  
- Yelp Academic Dataset (Kaggle): https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset

Mapping (Yelp):
- 1–2 stars → negative  
- 3 stars → neutral  
- 4–5 stars → positive

## Project structure

- Sentiment_Analysis.ipynb — Full BERT training & evaluation notebook  
- amazon_sentiment_bert/ — saved model folder

## Preprocessing

Common preprocessing steps:
- URL removal  
- Punctuation normalization  
- Lowercasing  
- Whitespace cleaning  
- Remove very short reviews  
- Balance classes (80k each)

## Training configuration

- Model: `bert-base-uncased`  
- Learning rate: 2e-5  
- Batch size: 16  
- Epochs: 6 (early stopping enabled)  
- Scheduler: Cosine LR with warmup steps (warmup_steps=1000)  
- Mixed precision: FP16  
- Optimization: AdamW

## Evaluation & Metrics

- Metrics: Accuracy, Precision / Recall / F1 (per-class & weighted), Confusion matrix  
- In-domain (Amazon) example results:  
  - Accuracy ≈ 0.77  
  - Weighted F1 ≈ 0.77  
- Out-of-domain (Yelp) example results:  
  - Yelp Accuracy ≈ 0.88 (on sampled 500 reviews)

Outputs include: classification reports, confusion matrices, example predictions.

## Usage

Load tokenizer & model (local folder):
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("./amazon_sentiment_bert")
model = AutoModelForSequenceClassification.from_pretrained("./amazon_sentiment_bert")

sentiment = pipeline("text-classification", model=model, tokenizer=tokenizer)
print(sentiment("This product is amazing!"))
```

Quick inference example:
- Returns label and score for each input string.

## Notebook

Open `Sentiment_Analysis.ipynb` to reproduce:
- Data loading & cleaning  
- Tokenization  
- Model fine-tuning  
- Saving/loading model  
- Evaluation on Amazon & Yelp  
- Visualizations: confusion matrices, classification reports, sample predictions

## Notes

- Ensure Hugging Face `transformers` and `datasets` are installed.  
- GPU recommended for training (mixed precision supported).
