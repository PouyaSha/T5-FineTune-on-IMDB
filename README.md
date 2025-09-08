# Sentiment Classification with T5 on IMDB Dataset

This repository contains a simple implementation of sentiment classification on the **IMDB movie reviews dataset** using a **T5-small model**. The code is adapted from the **Homework 1 notebooks of the LLM course by Prof. Rohban et al.**, where we explored different fine-tuning strategies for large language models (LLMs).

---

## Overview

We experimented with three main scenarios:

1. **Zero-Shot - unconstrained output:**  
   The pre-trained T5 model predicts without any constraints. It can generate any text, not limited to "positive" or "negative".  
   - **Accuracy:** ~0% (random, meaningless outputs)  
   - **Example predictions:**
   - Sample 1:
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
  
     

