# Sentiment Classification with T5 on IMDB Dataset

This repository contains a simple implementation of sentiment classification on the **IMDB movie reviews dataset** using a **T5-small model**. The code is adapted from the **Homework 1 notebooks of the LLM course by Prof. Rohban et al.**, where we explored different fine-tuning strategies for large language models (LLMs).

---

## Overview

We experimented with three main scenarios:

1. **Zero-Shot - unconstrained output:**  
   The pre-trained T5 model predicts without any constraints. It can generate any text, not limited to "positive" or "negative".  
   - **Accuracy:** ~0% (random, meaningless outputs)  
   - **Example predictions:**
     ```
      Sample 1:
      Review : worth the entertainment value of a rental, especially if you like action movies...
      True : negative
      Pred : the movie,
      
      Sample 2:
      Review : i turned over to this film in the middle of the night...
      True : positive
      Pred : : 8
      
      Sample 3:
      Review : to think this film was made the year i was born...
      True : positive
      Pred : punishment park is
      
      Sample 4:
      Review : "in the sweltering summer of 1958, the deuces, a gang of brooklyn toughs...
      True : negative
      Pred : a
      
      Sample 5:
      Review : i would have given this film a one star vote...
      True : negative
      Pred : a
     ```


2. **Zero-Shot - constrained to "negative" or "positive":**  
   Here we forced the model to only predict one of the two labels by comparing the **softmax probability of the first token** corresponding to "positive" and "negative".  
   - **Accuracy:** 75.55%  
   - **Example predictions:**
     ```
     Sample 1: positive
     Sample 2: positive
     Sample 3: negative
     Sample 4: negative
     Sample 5: positive
     ```

3. **Full Fine-Tuning (10 epochs):**  
   The T5 model is fine-tuned on the IMDB training dataset for 10 epochs using cross-entropy loss.  
   - **Accuracy:** 90.00%  
   - **Example predictions:**
     ```
     Sample 1: negative
     Sample 2: positive
     Sample 3: positive
     Sample 4: positive
     Sample 5: negative
     ```

---

## Training Results

The training and validation results for each epoch are summarized below:

| Epoch | Train Loss | Validation Accuracy |
|-------|------------|-------------------|
| 0     | 1.7465     | 0.83156           |
| 1     | 0.2042     | 0.85616           |
| 2     | 0.1766     | 0.86936           |
| 3     | 0.1634     | 0.87992           |
| 4     | 0.1546     | 0.88752           |
| 5     | 0.1462     | 0.89172           |
| 6     | 0.1410     | 0.89504           |
| 7     | 0.1363     | 0.89708           |
| 8     | 0.1328     | 0.89868           |
| 9     | 0.1279     | 0.90000           |

The **validation accuracy vs. epochs** plot is included below:

![Validation Accuracy vs. Epochs](images/valid_acc_plot.png)  
*(Make sure to save your plot as `valid_acc_plot.png` in an `images` folder in your repo.)*

---

## Jupyter Notebook

The full implementation is available in the Jupyter notebook file:  

**[LLM_Homework1_T5_IMDB.ipynb](notebooks/LLM_Homework1_T5_IMDB.ipynb)**  

This notebook contains:  
- Data preprocessing and tokenization using HuggingFace `datasets` and `T5TokenizerFast`  
- Zero-shot inference (unconstrained and forced to positive/negative)  
- Full fine-tuning of T5 for sentiment classification  
- Visualization of training loss and validation accuracy  

---

## How to Run

1. Clone this repository:
```bash
git clone https://github.com/<your-username>/t5-imdb-sentiment.git
cd t5-imdb-sentiment

     

