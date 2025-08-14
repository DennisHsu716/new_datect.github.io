### 📰 Fake News Detection AI
## 📌 Project Overview
This project demonstrates a fake news detection pipeline using a baseline TF-IDF + Logistic Regression model and an advanced RoBERTa fine-tuning approach with Hugging Face Transformers, with the dataset stored in Google Drive and processed in Google Colab
## 📂 Repository Structure
```
fake-news/  
│  
├── config/                  # Config files (optional)  
├── data/  
│   ├── raw/                 # Raw CSV dataset  
│   └── processed/           # Preprocessed JSONL dataset  
│  
├── src/  
│   ├── prepare.py           # Data cleaning & preprocessing  
│   ├── baseline.py          # TF-IDF + Logistic Regression  
│   └── transformer_ft.py    # Transformer fine-tuning (RoBERTa)  
│  
├── runs/                    # Model outputs & metrics  
│   ├── baseline.json  
│   └── roberta_metrics.json  
│
└── app/                     # (Optional) App integration
```

## 📂 Dataset Structure
```
data/
│── raw/
│   └── news.csv             # Contains text + label
│── processed/
│   └── dataset.jsonl
```
* text → News article content

* label → ```0``` for real news, ```1``` for fake news

## ⚙ Environment Setup
```
pip install -U pandas numpy scikit-learn transformers datasets
```
## 🚀 Steps to Run
1️⃣ Data Preparation
```
python src/prepare.py \
  --input data/raw/news.csv \
  --out data/processed/dataset.jsonl
```
2️⃣ Baseline Model (TF-IDF + Logistic Regression)
```
python src/baseline.py \
  --input data/processed/dataset.jsonl \
  --out runs/baseline.json
```
3️⃣ Transformer Fine-Tuning (RoBERTa)
```
python src/transformer_ft.py \
  --input data/processed/dataset.jsonl \
  --model_name roberta-base \
  --epochs 2 \
  --out runs/roberta_metrics.json
```
## 📊 Example Output

Baseline:
```
{"f1": 0.82}
```

Transformer:
```
{"eval_loss": 0.45, "eval_accuracy": 0.87, "eval_f1": 0.85}
```
## 📌 Future Improvements

* Increase dataset size for better model performance

* Apply advanced text augmentation techniques

* Experiment with larger models (e.g., roberta-large, deberta-v3-base)
