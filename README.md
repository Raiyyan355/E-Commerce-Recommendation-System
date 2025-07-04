# E-Commerce Product Recommendation System

A Flask-based Machine Learning web application that provides intelligent product recommendations using content-based, collaborative filtering, and hybrid recommendation techniques.

## 📌 Project Overview

This project helps users discover relevant products from an e-commerce catalog by entering the name of a product they like. The system generates similar product suggestions using a **hybrid recommendation approach**.

### 🔍 Features
- Content-Based Filtering using **TF-IDF** and **Cosine Similarity**
- Simplified Collaborative Filtering based on user-product interaction matrix
- A Hybrid model that combines both approaches
- Clean and interactive Flask web interface
- Product display with image, rating, brand, and review count
- Input options for selecting product and number of recommendations

---

## 🧠 Recommendation Techniques

### ✅ 1. Content-Based Filtering
- Uses product **tags** to compute item similarity via TF-IDF.
- Compares the selected product with others to find similar items.

### ✅ 2. Collaborative Filtering (Simplified)
- Generates a **user-product rating matrix**.
- Computes user similarity using **cosine similarity**.
- Recommends products liked by similar users that the target user hasn’t rated.

### ✅ 3. Hybrid Recommendations
- Merges results from both content-based and collaborative filtering.
- Removes duplicates and ranks top-N recommendations.

---

## ⚙️ Tech Stack

| Technology      | Usage                           |
|----------------|----------------------------------|
| Python          | Core programming language        |
| Flask           | Web framework for deployment     |
| Pandas          | Data manipulation                |
| NumPy           | Numerical operations             |
| scikit-learn    | TF-IDF vectorization, similarity |
| HTML/CSS        | Web UI templates                 |

---

## 📂 Dataset

- Used a sample `.tsv` file (`marketing_sample.tsv`) containing:
  - Product ID, Name, Description, Category, Brand, Rating, Review Count, Tags, and Image URL.
- Basic preprocessing: renaming columns, filling missing values, extracting numeric IDs.

---

## 🚀 How to Run the Project

### Prerequisites:
- Python 3.7+
- Flask
- Required packages: `pandas`, `numpy`, `scikit-learn`

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Raiyyan355/E-Commerce-Recommendation-System.git
   cd E-Commerce-Recommendation-System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place `marketing_sample.tsv` in the root directory.

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 🌐 Application Interface

- Dropdown to select any product name from the dataset.
- Input field to select number of recommendations (default: 10).
- Displays cards with:
  - Product Name
  - Image
  - Brand
  - Rating
  - Review Count

---

## 📌 Limitations & Future Enhancements

- Currently uses simplified collaborative filtering without explicit user history.
- User IDs are simulated — no actual login or profile integration.
- Could be extended with:
  - Real-time user feedback (likes/dislikes)
  - Deep learning models
  - Personalization with login sessions
  - Product search bar and filter options

---


## 📁 Folder Structure

```
├── app.py                  # Main Flask application
├── marketing_sample.tsv    # Dataset file (to be added)
├── templates/
│   └── index.html          # Main web page
├── static/
│   └── css/ / images/      # Optional styling or media
└── README.md               # This documentation
```
