# 🎬 IMDB Sentiment Classification with LSTM and PyTorch

> **Binary sentiment analysis on movie reviews using a custom LSTM neural network built from scratch in PyTorch — no pretrained models, no shortcuts.**


## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [NLP Preprocessing Pipeline](#nlp-preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

---

## Overview

This project implements **sentiment analysis** on the IMDB movie review dataset using a **Long Short-Term Memory (LSTM)** recurrent neural network. The goal is to classify each movie review as either **positive** or **negative** sentiment — a classic Natural Language Processing (NLP) binary classification task.

The entire pipeline is built from scratch, including:
- A custom tokenizer and vocabulary builder
- Manual word-to-index encoding/decoding
- Padded and packed sequence handling
- An LSTM-based deep learning model trained end-to-end

This project is ideal for anyone learning **deep learning for NLP**, **PyTorch sequence modelling**, or **text classification** techniques.

---

## Features

- ✅ Full NLP preprocessing pipeline (lowercasing, stopword removal, lemmatization)
- ✅ Custom vocabulary builder with `<PAD>` and `<UNK>` tokens
- ✅ Efficient sequence padding with `pad_sequence` and `pack_padded_sequence`
- ✅ LSTM-based sentiment classifier with dropout regularization
- ✅ GPU support via CUDA
- ✅ Model saving and loading with `torch.save` / `torch.load`
- ✅ Exploratory Data Analysis (EDA) with Matplotlib and Seaborn
- ✅ Reproducible results via fixed random seeds

---

## Architecture

```
Raw Text Review
      │
      ▼
NLP Preprocessing (lowercase → stopword removal → lemmatization)
      │
      ▼
Custom Vocabulary (vocab dict with <PAD>=0, <UNK>=1)
      │
      ▼
Tokenization & Integer Encoding (max length = 256)
      │
      ▼
Padding / Packing (pad_sequence → pack_padded_sequence)
      │
      ▼
Embedding Layer (vocab_size → 256 dims)
      │
      ▼
LSTM Layer (256 hidden units)
      │
      ▼
Dropout (p=0.3)
      │
      ▼
Fully Connected Layer (256 → 1)
      │
      ▼
Sigmoid Activation → Binary Prediction (0: Negative, 1: Positive)
```

---

## Dataset

- **Source:** [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/rehanliaqat17/imbd-dataset)
- **Size:** 50,000 movie reviews (25,000 positive, 25,000 negative)
- **Split:** 80% training / 20% testing
- **Format:** CSV with `review` (text) and `sentiment` (positive/negative) columns

The dataset is downloaded automatically using `kagglehub`.

---

## NLP Preprocessing Pipeline

Each review goes through the following steps before being fed to the model:

1. **Lowercasing** — Normalise text case
2. **Stopword Removal** — Remove common English words (e.g. "the", "is") using NLTK
3. **Punctuation Removal** — Strip all punctuation characters
4. **POS Tagging** — Identify part-of-speech for accurate lemmatization
5. **Lemmatization** — Reduce words to their base/root form using `WordNetLemmatizer`
6. **Vocabulary Construction** — Build a word-to-index mapping with special tokens
7. **Integer Encoding** — Convert tokens to integer sequences
8. **Padding** — Pad/truncate sequences to a fixed max length of 256

---

## Model Architecture

```python
class SentimentModel(nn.Module):
    def __init__(self, vocab_size):
        self.embedding = nn.Embedding(vocab_size, embedding_dim=256, padding_idx=0)
        self.lstm      = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(256, 1)
        self.sigmoid   = nn.Sigmoid()
```

| Layer | Output Shape | Notes |
|---|---|---|
| Embedding | `(batch, seq_len, 256)` | Learned word representations |
| LSTM | `(batch, 256)` | Takes last hidden state |
| Dropout | `(batch, 256)` | p=0.3, reduces overfitting |
| Linear | `(batch, 1)` | Single output neuron |
| Sigmoid | `(batch, 1)` | Outputs probability [0, 1] |

---

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy (BCELoss) |
| Batch Size | 64 |
| Epochs | 10 |
| Max Sequence Length | 256 |
| Gradient Clipping | 1.0 |
| Random Seed | 42 |

---

## Results

The model is evaluated on a held-out test set (20% of the dataset) after each epoch. Final metrics are printed at the end of training.

> Run the notebook yourself to see the full training logs and final test accuracy.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/imdb-sentiment-lstm.git
cd imdb-sentiment-lstm

# Install dependencies
pip install torch torchvision nltk pandas scikit-learn matplotlib seaborn kagglehub
```

Download the required NLTK data (handled automatically in the notebook):

```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('punkt')
```

---

## Usage

1. Open the notebook:
   ```bash
   jupyter notebook 6_IMBD_classification.ipynb
   ```

2. Run all cells from top to bottom.

3. The trained model will be saved as `sentiment_model.pth` in the project directory.

4. To reload and use the saved model:
   ```python
   model = SentimentModel(vocab_size)
   model.load_state_dict(torch.load("sentiment_model.pth"))
   model.eval()
   ```

---

## Project Structure

```
imdb-sentiment-lstm/
│
├── 6_IMBD_classification.ipynb   # Main Jupyter Notebook
├── sentiment_model.pth            # Saved model weights (after training)
└── README.md                      # This file
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| **PyTorch** | Deep learning framework, LSTM model |
| **NLTK** | NLP preprocessing (stopwords, lemmatization, POS tagging) |
| **pandas** | Data loading and manipulation |
| **scikit-learn** | Train/test split |
| **Matplotlib / Seaborn** | Exploratory Data Analysis (EDA) |
| **KaggleHub** | Automated dataset download |

---

## Keywords

`sentiment analysis` · `IMDB dataset` · `LSTM` · `PyTorch` · `NLP` · `text classification` · `binary classification` · `deep learning` · `natural language processing` · `movie review classification` · `recurrent neural network` · `RNN` · `sequence modelling` · `word embeddings` · `lemmatization` · `stopword removal`

---

## Author

**Arian Latifi** — Computer Science Student  
Feel free to open an issue or submit a pull request if you have suggestions or improvements!

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
