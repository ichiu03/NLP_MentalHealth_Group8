# Reddit Depression Detection — NLP Final Project

Binary classification of Reddit posts as depressed vs. non-depressed, using a shared prepared dataset and three modeling approaches: a TF-IDF + Logistic Regression baseline, BERT fine-tuning, and LoRA fine-tuning using open-source Mistral AI Model.

---

## Directory Structure

```
NLP_MENTALHEALTH_GROUP8/
├── figures/                          
│   ├── class_balance_before_after_filter.png
│   ├── class_balance_before_after_balancing.png
│   ├── word_count_before_after_matching_v1.png
│   ├── word_count_before_after_matching_v2.png
│   ├── dataset_split_sizes.png
│   ├── baseline_model_comparison.png
│   ├── confusion_matrix.png
│   └── wordclouds.png
├── NLPDataSetup_Baseline.ipynb       # data prep + TF-IDF baseline 
├── BERT_Mental_Health_Detection_NLP-3.ipynb         
└── Mistral_7B_Mental_Health_Testing.ipynb
```

---

## Where to Find the Code

| What | File |
|---|---|
| Data preparation, cleaning, balancing, splitting | `NLPDataSetup_Baseline.ipynb` |
| TF-IDF + Logistic Regression baseline | `NLPDataSetup_Baseline.ipynb` (bottom half) |
| BERT fine-tuning | `BERT_Mental_Health_Detection_NLP-3.ipynb` |
| LoRA fine-tuning | `Mistral_7B_Mental_Health_Testing.ipynb` |



## How to Run the Code

All notebooks run in Google Colab. Open in Colab, upload dataset to Drive, mount your Drive when prompted, and run cells top to bottom.

**Order matters:**

1. **`NLPDataSetup_Baseline.ipynb`** — run this first. It generates the four CSV splits that every other notebook depends on.
2. **`BERT_Mental_Health_Detection_NLP-3.ipynb`** — loads from the saved CSVs, can be run independently after step 1.
3. **`Mistral_7B_Mental_Health_Testing.ipynb`** — loads from the saved CSVs, can be run independently after step 1.

---

## Data Source

- **Reddit Depression Dataset** — sourced from Kaggle:
  https://www.kaggle.com/datasets/rishabhkausish/reddit-depression-dataset

---

## Non-Standard Libraries

Libraries beyond the standard ML stack worth noting:

- **`wordcloud`** — used for per-class word cloud visualizations
  https://github.com/amueller/word_cloud
  Install: `pip install wordcloud`

---

## Notes on Data Preparation

The raw dataset has a significant class imbalance (~4x more non-depressed posts). The preparation pipeline in `NLPDataSetup_Baseline.ipynb` does the following:

1. Combines `title` and `body` columns into a single `text` column
2. Drops posts under 10 words (after combining)
3. Balances classes using **histogram-matched downsampling** — non-depressed posts are sampled bin-by-bin to mirror the word-count distribution of depressed posts, then trimmed to a 1:1 ratio. This avoids introducing a confound where one class systematically has longer posts.
4. Splits into train/val/test (70/15/15), stratified by label

---

## BERT Fine-Tuning

**Model used:**
`bert-base-uncased` pretrained model fine-tuned for binary sequence classification
using HuggingFace's `BertForSequenceClassification` with a linear classification
head on top (2 output labels: depressed / non-depressed).
- Model card: [bert-base-uncased](https://huggingface.co/bert-base-uncased)
- Architecture: [BertForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification)

**Non-standard libraries:**
- [`transformers`](https://huggingface.co/docs/transformers) — BERT tokenizer and model
- [`torch`](https://pytorch.org/) — model training, DataLoader, loss function
- [`scikit-learn`](https://scikit-learn.org/) — evaluation metrics

**Results on test set:**
| Metric | Score |
|---|---|
| Accuracy | 95.93% |
| Macro F1 | 0.90 |
| Precision (Depressed) | 0.87 |
| Recall (Depressed) | 0.92 |

## LoRA Fine-Tuning

**Model used:**
`mistralai/Mistral-7B-v0.3` pretrained model fine-tuned for binary classification using Unsloth's `FastLanguageModel` with LoRA adapters (r=16) for parameter-efficient training (2 output labels: depressed / non-depressed).
- Model card: [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) via [unsloth/mistral-7b-v0.3-bnb-4bit](https://huggingface.co/unsloth/mistral-7b-v0.3-bnb-4bit)
- LoRA framework: [Unsloth](https://unsloth.ai/docs)

**Non-standard libraries / APIs:**
- [`unsloth`](https://unsloth.ai/docs) — LoRA fine tuning, Mistral 7B inference
- [`trl`](https://huggingface.co/docs/trl/index) — supervised fine-tuning using SFTTrainer
- [`transformers`](https://huggingface.co/docs/transformers) — model tokenizer, training configuration
- [`scikit-learn`](https://scikit-learn.org/) — evaluation metrics
- [`tqdm`](https://pypi.org/project/tqdm/) — progress tracker for long runtimes

**Training approach**
Fine tuned Mistral 7B using LoRA (Low-Rank Adaptation) to adjust pre-loaded weights. Training text was masked using `SFTTrainer` to optimize only on classification label rather than full input text. Model output single character response of `1` or `0`. 

Training was done with batch size of 2, gradient accumulation of 4 steps, learning rate of 2e-4, and 3 epochs.

Tested model after 0, 2, 100, 1,000, 10,000 rows of training, resetting model in between trainings. Number of validation rows during training equal to 10% of training rows.

**Results on test set:**
| Metric | No Training | 2 Rows | 100 Rows | 1,000 Rows | 10,000 Rows |
|---|---|---|---|---|---|
| Accuracy | 40% | 40% | 59% | 92% | 94% |
| F1 Score | 0.37 | 0.37 | 0.56 | 0.92 | 0.94 |
| Precision | 0.39 | 0.39 | 0.59 | 0.92 | 0.94 |
| Recall | 0.42 | 0.42 | 0.57 | 0.92 | 0.94 |
