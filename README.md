# DecodeLabs Internship — Tasks

This repository contains my submitted projects for the DecodeLabs Artificial Intelligence Internship (Batch 2026).

**Intern:** Muhammad Anas
**Domain:** Artificial Intelligence

---

## 📁 Projects

### 1. [Rule-Based Chatbot](./project1_chatbot)
A simple rule-based chatbot built using conditional logic and pattern matching to respond to user queries.

**Skills:** Logic building, conditional statements, string matching

---

### 2. [Iris Flower Classification](./Project2_Iris_Classification)
A machine learning classification project that predicts the species of an Iris flower based on its petal and sepal measurements, using the classic Iris dataset.

**Skills:** Data preprocessing, classification algorithms, model evaluation
**Tool used:** Google Colab

---

### 3. [Tech Stack Recommender](./Project3_TechStackRecommender)
An AI-powered recommendation engine that suggests the most relevant job roles based on a user's skills, using **Content-Based Filtering**.

**How it works:**
- Takes 3+ user skills as input
- Converts skills into **TF-IDF vectors**
- Computes **Cosine Similarity** between the user profile and 218 real job roles (sourced from a Kaggle dataset)
- Returns the **Top 3** best-matching job roles with match scores

**Tech stack:** Python, pandas, scikit-learn (TfidfVectorizer, cosine_similarity)

**How to run:**
```bash
pip install pandas scikit-learn
python tech_stack_recommender.py
```

---

## 🏫 About

These projects were completed as part of the **DecodeLabs Industrial Training Kit** — a hands-on internship program focused on practical AI engineering skills, including logic building, pattern matching, machine learning fundamentals, and recommendation system design.
