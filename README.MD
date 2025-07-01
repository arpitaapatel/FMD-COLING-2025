# 🧠 Financial Misinformation Detection 

This project aims to detect misinformation in financial claims using various machine learning and deep learning models including Logistic Regression, RNN, and BERT.

---

## 📁 Dataset

- **Name**: COLING25-FMD
- **Source**: [Hugging Face](https://huggingface.co/datasets/lzw1008/COLING25-FMD)
- **Classes**:
  - ✅ `True`: Claim is factually correct
  - ❌ `False`: Claim is factually incorrect
  - ⚠️ `NEI`: Not enough information to verify

---

## 🧪 Models Used

| Model               | Preprocessing                         | Accuracy | Precision | Recall |
|--------------------|----------------------------------------|----------|-----------|--------|
| Logistic Regression| TF-IDF vectorization                   | 65%      | 66.2%     | 65 %   |
| RNN (LSTM)         | Tokenized + padded text sequences      | 60%      | 65%       | 60%    |
| BERT               | Pretrained BERT + fine-tuning          | 50%      | 25%       | 60%    |


---

## 🛠️ Preprocessing Steps

1. Text Cleaning (lowercasing, stopword removal)
2. Lemmatization
3. Label encoding (`True`, `False`, `NEI`)
4. Tokenization (TF-IDF for LR, Tokenizer + Padding for RNN, BERT tokenizer)

---

## 🧠 Model Training & Evaluation

- Logistic Regression using Scikit-learn
- RNN using Keras (LSTM layers)
- BERT using HuggingFace Transformers (`TFBertForSequenceClassification`)
- Evaluation based on Accuracy, Precision, and Recall

---

## 📊 Results

Final comparison shows that **BERT** outperforms traditional models in all metrics, especially in handling ambiguous (NEI) cases.

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/gauransh-juneja/your-repo-name.git

# Navigate to the directory
cd your-repo-name

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install requirements
pip install -r requirements.txt

# Run the notebook
jupyter notebook
