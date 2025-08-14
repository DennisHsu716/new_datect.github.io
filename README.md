### 📰 Fake News Detection AI
## 📌 Project Overview
This project demonstrates a fake news detection pipeline using a baseline TF-IDF + Logistic Regression model and an advanced RoBERTa fine-tuning approach with Hugging Face Transformers, with the dataset stored in Google Drive and processed in Google Colab
## 📂 Repository Structure
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

## 📂 Dataset Structure
```
data/
│── raw/
│   └── news.csv             # Contains text + label
│── processed/
│   └── dataset.jsonl
```

