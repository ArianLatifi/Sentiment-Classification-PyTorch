# Text Classification with LSTM and PyTorch

> **Two end-to-end NLP classification projects built from scratch in PyTorch — binary sentiment analysis on IMDB movie reviews and multi-class news category classification on the HuffPost dataset — using custom LSTM neural networks, no pretrained models.**

This repository contains two standalone Jupyter notebooks that demonstrate text classification using Long Short-Term Memory (LSTM) recurrent neural networks in PyTorch. Both projects share a common NLP preprocessing pipeline and are designed to teach practical deep learning for natural language processing.

---

## Table of Contents

- [Projects Overview](#projects-overview)
- [Project 1: IMDB Sentiment Analysis](#project-1-imdb-sentiment-analysis)
  - [Dataset](#dataset-imdb)
  - [Preprocessing Pipeline](#preprocessing-pipeline-imdb)
  - [Model Architecture](#model-architecture-imdb)
  - [Training Configuration](#training-configuration-imdb)
  - [Results](#results-imdb)
- [Project 2: News Category Classification](#project-2-news-category-classification)
  - [Dataset](#dataset-news)
  - [Category Mapping](#category-mapping)
  - [Preprocessing Pipeline](#preprocessing-pipeline-news)
  - [Model Architecture](#model-architecture-news)
  - [Training Configuration](#training-configuration-news)
  - [Results](#results-news)
- [Shared NLP Preprocessing Pipeline](#shared-nlp-preprocessing-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Keywords](#keywords)

---

## Projects Overview

| | Project 1 | Project 2 |
|---|---|---|
| **Notebook** | `IMBD_classification.ipynb` | `News_Classifier_LSTM.ipynb` |
| **Task** | Binary Sentiment Classification | Multi-class News Categorization |
| **Dataset** | IMDB Movie Reviews (50K) | HuffPost News Articles (209K) |
| **Classes** | 2 (positive / negative) | 11 (politics, lifestyle, health, ...) |
| **Loss Function** | BCELoss | CrossEntropyLoss |
| **Test Accuracy** | **87.45%** | **68%** |

---

## Project 1: IMDB Sentiment Analysis

Binary sentiment classification on 50,000 IMDB movie reviews. The model learns to distinguish positive from negative reviews using a custom-built vocabulary and an LSTM classifier trained end-to-end.

### Pipeline Overview

```
Raw Movie Review Text
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

### Dataset (IMDB)

- **Source:** [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/rehanliaqat17/imbd-dataset)
- **Size:** 50,000 movie reviews — 25,000 positive, 25,000 negative (perfectly balanced)
- **Split:** 80% training (40,000) / 20% testing (10,000)
- **Format:** CSV with `review` (text) and `sentiment` (positive/negative) columns
- **Download:** Handled automatically via `kagglehub`

### Preprocessing Pipeline (IMDB)

Each review is processed through these steps before encoding:

1. **Lowercasing** — Normalize text case
2. **Stopword Removal** — Remove common English words using NLTK
3. **Punctuation Removal** — Strip all punctuation characters
4. **POS Tagging** — Identify part-of-speech for accurate lemmatization
5. **Lemmatization** — Reduce words to their root form using `WordNetLemmatizer`
6. **Vocabulary Construction** — Build word-to-index mapping with `<PAD>` and `<UNK>` tokens
7. **Integer Encoding** — Convert token sequences to integer sequences
8. **Padding / Truncation** — Fixed max sequence length of 256 tokens

### Model Architecture (IMDB)

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

### Training Configuration (IMDB)

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | BCELoss (Binary Cross-Entropy) |
| Batch Size | 64 |
| Epochs | 10 |
| Max Sequence Length | 256 |
| Gradient Clipping | 1.0 |
| Random Seed | 42 |

### Results (IMDB)

| Epoch | Train Loss | Test Accuracy |
|---|---|---|
| 1 | 0.4903 | 80.55% |
| 2 | 0.3098 | 87.26% |
| 3 | 0.2029 | 86.85% |
| 5 | 0.0671 | 87.28% |
| 10 | 0.0122 | 87.45% |

**Final Test Accuracy: 87.45%** | **Final Test Loss: 0.6646**

---

## Project 2: News Category Classification

Multi-class text classification on 209,527 HuffPost news articles. The model classifies each article headline + description into one of 11 consolidated news categories using an LSTM network with CrossEntropyLoss.

### Pipeline Overview

```
News Headline + Short Description
      │
      ▼
NLP Preprocessing (lowercase → stopword removal → lemmatization)
      │
      ▼
Custom Vocabulary (vocab dict with <PAD>=0, <UNK>=1, vocab cached as .pth)
      │
      ▼
Tokenization & Integer Encoding (max length = 100)
      │
      ▼
Padding / Packing (pad_sequence → pack_padded_sequence)
      │
      ▼
Embedding Layer (vocab_size → 32 dims)
      │
      ▼
LSTM Layer (32 → 128 hidden units)
      │
      ▼
Fully Connected Layer (128 → 11 classes)
      │
      ▼
Softmax → Multi-class Prediction (11 news categories)
```

### Dataset (News)

- **Source:** [HuffPost News Category Dataset v3 on Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- **Size:** 209,527 news articles
- **Format:** JSON Lines with `headline`, `short_description`, `category`, `authors`, `date`, and `link`
- **Input text:** Concatenation of `headline` and `short_description`
- **Download:** Handled automatically via `kagglehub`

### Category Mapping

The original 42 fine-grained categories are merged into 11 consolidated classes:

| Consolidated Label | Original Categories | Count |
|---|---|---|
| `lifestyle` | TRAVEL, STYLE & BEAUTY, STYLE, HOME & LIVING, WEDDINGS, FOOD & DRINK, TASTE, PARENTING, PARENTS | 51,123 |
| `politics` | POLITICS, WORLD NEWS, THE WORLDPOST, WORLDPOST, U.S. NEWS | 46,521 |
| `entertainment` | ENTERTAINMENT, COMEDY, ARTS, ARTS & CULTURE, CULTURE & ARTS | 26,684 |
| `health` | WELLNESS, HEALTHY LIVING | 24,639 |
| `society` | WOMEN, BLACK VOICES, LATINO VOICES, QUEER VOICES, RELIGION, COLLEGE, EDUCATION, DIVORCE | 23,793 |
| `business_tech` | BUSINESS, MONEY, TECH | 9,852 |
| `misc` | WEIRD NEWS, GOOD NEWS, IMPACT, FIFTY | 9,060 |
| `science_environment` | SCIENCE, GREEN, ENVIRONMENT | 6,272 |
| `sports` | SPORTS | 5,077 |
| `crime` | CRIME | 3,562 |
| `media` | MEDIA | 2,944 |

### Preprocessing Pipeline (News)

Identical pipeline to Project 1, with these differences:

- **Max sequence length:** 100 tokens (shorter, as headlines are brief)
- **Input text:** `headline + " " + short_description` concatenated before preprocessing
- **Vocabulary caching:** Saved to disk as `vocabs.pth` to avoid recomputation on reruns
- Empty texts after preprocessing default to `["<PAD>"]`

### Model Architecture (News)

```python
class NewsCategorizer(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = torch.nn.LSTM(embed_dim, hidden_size=128, batch_first=True)
        self.fc        = torch.nn.Linear(128, num_classes)
```

| Layer | Output Shape | Notes |
|---|---|---|
| Embedding | `(batch, seq_len, 32)` | Compact 32-dim representations |
| LSTM | `(batch, 128)` | Takes last hidden state |
| Linear | `(batch, 11)` | One logit per news category |
| (CrossEntropyLoss applies softmax internally) | | |

### Training Configuration (News)

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.01 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Epochs | 10 |
| Max Sequence Length | 100 |
| Random Seed | 42 |
| Vocabulary Size | ~107,791 tokens |

### Results (News)

| Epoch | Train Loss | Train Accuracy |
|---|---|---|
| 1 | 6250.84 | 62.04% |
| 2 | 4755.55 | 71.17% |
| 5 | 3706.80 | 77.39% |
| 8 | 3282.10 | 80.01% |
| 10 | 3139.28 | 80.85% |

**Final Test Accuracy: 68%** (across 41,906 test samples)

Per-class performance on the test set:

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| politics | 0.76 | 0.81 | 0.78 |
| entertainment | 0.67 | 0.64 | 0.66 |
| lifestyle | 0.74 | 0.82 | **0.78** |
| business_tech | 0.52 | 0.52 | 0.52 |
| sports | 0.61 | 0.62 | 0.62 |
| misc | 0.30 | 0.24 | 0.26 |
| science_environment | 0.47 | 0.46 | 0.47 |
| society | 0.62 | 0.55 | 0.58 |
| crime | 0.45 | 0.43 | 0.44 |
| health | 0.74 | 0.67 | 0.70 |
| media | 0.60 | 0.30 | 0.40 |

---

## Shared NLP Preprocessing Pipeline

Both projects use the same preprocessing functions:

```python
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
    return tag_map.get(tag, wordnet.NOUN)

def preprocess_text(text: str):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ' '.join([w for w in text.split() if w not in stop_words])
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text.split()])
    return text.split()
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ArianLatifi/Classification-PyTorch.git
cd Classification-PyTorch

# Install dependencies
pip install torch torchvision nltk pandas numpy scikit-learn matplotlib seaborn kagglehub
```

Download the required NLTK data (handled automatically inside each notebook):

```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('punkt')
```

---

## Usage

### Run the IMDB Sentiment Classifier

```bash
jupyter notebook IMBD_classification.ipynb
```

1. Run all cells from top to bottom.
2. The trained model is saved as `sentiment_model.pth`.
3. To reload the saved model:

```python
model = SentimentModel(vocab_size)
model.load_state_dict(torch.load("sentiment_model.pth"))
model.eval()
```

### Run the News Category Classifier

```bash
jupyter notebook News_Classifier_LSTM.ipynb
```

1. Run all cells from top to bottom.
2. The vocabulary is cached as `vocabs.pth` on the first run to speed up reruns.
3. The trained model is saved as `News.pth`.
4. To reload the saved model:

```python
vocab = torch.load("../asset/7_news/vocabs.pth")
model = NewsCategorizer(len(vocab), embed_dim=32, num_classes=11)
model.load_state_dict(torch.load("../asset/7_news/News.pth"))
model.eval()
```

---

## Project Structure

```
Classification-PyTorch/
│
├── IMBD_classification.ipynb      # Project 1: Binary sentiment analysis (IMDB)
├── News_Classifier_LSTM.ipynb     # Project 2: Multi-class news categorization
├── sentiment_model.pth            # Saved IMDB model weights (generated after training)
└── README.md                      # This file
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| **PyTorch** | LSTM model definition, training loop, GPU acceleration |
| **NLTK** | Tokenization, stopword removal, POS tagging, lemmatization |
| **pandas** | Data loading, manipulation, and label encoding |
| **NumPy** | Numerical operations and array handling |
| **scikit-learn** | Train/test split, confusion matrix, classification report |
| **Matplotlib / Seaborn** | EDA plots, confusion matrix heatmap |
| **KaggleHub** | Automated dataset download from Kaggle |

---

## Keywords

`text classification` · `LSTM` · `PyTorch` · `NLP` · `natural language processing` · `sentiment analysis` · `IMDB dataset` · `movie review classification` · `binary classification` · `news classification` · `news category prediction` · `multi-class classification` · `HuffPost dataset` · `deep learning` · `recurrent neural network` · `RNN` · `sequence modelling` · `word embeddings` · `lemmatization` · `stopword removal` · `POS tagging` · `custom vocabulary` · `pack_padded_sequence` · `CrossEntropyLoss` · `BCELoss` · `text preprocessing pipeline` · `Kaggle NLP`

---

## Author

**Arian Latifi** — Computer Science Student  
Feel free to open an issue or submit a pull request if you have suggestions or improvements!
