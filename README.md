# Reddit Depression Detection — NLP Final Project

Binary classification of Reddit posts as depressed vs. non-depressed, using a shared prepared dataset and three modeling approaches: a TF-IDF + Logistic Regression baseline, BERT fine-tuning, and GPT prompting.

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
├── BERT_Mental_Health_Detection_NLP
├── BERT_Mental_Health_Detection_NLP-3         
└── GPT.ipynb               
```

---

## Where to Find the Code

| What | File |
|---|---|
| Data preparation, cleaning, balancing, splitting | `NLPDataSetup_Baseline.ipynb` |
| TF-IDF + Logistic Regression baseline | `NLPDataSetup_Baseline.ipynb` (bottom half) |
| BERT fine-tuning | `BERT_Classification.ipynb` |
| GPT prompting | `GPT_Prompting.ipynb` |



## How to Run the Code

All notebooks run in Google Colab. Open in Colab, upload dataset to Drive, mount your Drive when prompted, and run cells top to bottom.

**Order matters:**

1. **`NLPDataSetup_Baseline.ipynb`** — run this first. It generates the four CSV splits that every other notebook depends on.
2. **`BERT_Classification.ipynb`** — loads from the saved CSVs, can be run independently after step 1.
3. **`GPT_Prompting.ipynb`** — loads from the saved CSVs, can be run independently after step 1.

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

## BERT Fine-Tuning (Teammate Section)

*To be filled in by teammate.*

**Model used:**
*(e.g., `bert-base-uncased` from HuggingFace — link here)*

**Non-standard libraries:**
*(e.g., `transformers`, `datasets` — links here)*

**Any notebooks or tutorials referenced:**
*(links here)*

**Results on test set:**
*(accuracy, macro F1)*

---

## GPT Prompting (Teammate Section)

*To be filled in by teammate.*

**Model used:**
*(e.g., `gpt-4o` via OpenAI API — link here)*

**Non-standard libraries / APIs:**
*(e.g., `openai` Python library — link here)*

**Prompting approach:**
*(zero-shot, few-shot, chain-of-thought, etc.)*

**Any notebooks or tutorials referenced:**
*(links here)*

**Results on test set:**
*(accuracy, macro F1)*
