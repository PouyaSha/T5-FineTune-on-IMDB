# Sentiment Classification with T5 on IMDB Dataset

This repository contains a simple implementation of sentiment classification on the **IMDB movie reviews dataset** using a **T5-small model**. The code is adapted from the **Homework 1 notebooks of the LLM course by Prof. Rohban et al.**, where we explored different fine-tuning strategies for large language models (LLMs).

---

## Overview

We experimented with three main scenarios:

1. **Zero-Shot - unconstrained output:**  
   The pre-trained T5 model predicts without any constraints. It can generate any text, not limited to "positive" or "negative".  
   - **Accuracy:** ~0% (random, meaningless outputs)  
   - **Example predictions:**

