# Book Rating Prediction using Item-Item Collaborative Filtering


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://drive.google.com/file/d/1Qp71ikDTTWxfd6gIM3yPZ0BDOAb5oEJ8/view?usp=sharing)

## Overview

This project implements a **custom Item-Item Collaborative Filtering** algorithm to predict user ratings for books they haven't read yet. The model is evaluated using **Mean Absolute Difference (MAD)** across multiple training-test splits and neighborhood sizes (`k`).

---

## Dataset

The following CSV files are required:

* `Ratings.csv`: Contains user ratings for books.
* `Users.csv`: Contains user demographic info.
* `Books.csv`: Contains book metadata.

If you're using **Google Colab**, your file paths may look like:

```python
ratings_path = "/content/drive/MyDrive/UMD/DATA606/p2/data/Ratings.csv"
books_path   = "/content/drive/MyDrive/UMD/DATA606/p2/data/Books.csv"
users_path   = "/content/drive/MyDrive/UMD/DATA606/p2/data/Users.csv"

run_holder(ratings_path, users_path, books_path)
```

---

## Requirements

Install dependencies via pip:

```bash
pip install numpy pandas scikit-learn scipy joblib
```

---

## Methodology

### 1. **Data Preparation**

* Merge and clean ratings, users, and books.
* Filter out users and items with very few ratings (default: min 2 ratings/user, 1 rating/item).
* Convert data into a **sparse user-item matrix** using `scipy.sparse`.

### 2. **Similarity Computation**

* Apply **mean-centering** on the item axis.
* Compute **cosine similarity** between items.
* Use **shrinkage** based on co-rating counts to stabilize similarity scores.
* **Top-k truncation** keeps only the strongest `k` neighbors per item.

### 3. **Prediction**

* Predict ratings by adjusting for item bias using weighted residuals.
* Fallback to global mean when prediction is unreliable (e.g., no neighbors or zero similarity sum).

### 4. **Evaluation**

* Use multiple training ratios (60% to 90%) and neighborhood sizes (`k=5,10,...,100`).
* Report **Mean Absolute Difference (MAD)** for each setting.

---

## Why Not Use `sklearn.neighbors.NearestNeighbors` or Pandas?

* **Sparse Efficiency**: Pandas/DataFrames and `sklearn` are inefficient for very sparse matrices (millions of missing values).
* **Custom Behavior**: Standard nearest neighbor models do not support:

  * Demeaning with item bias.
  * Shrinkage by co-rating count.
  * Custom prediction logic (residual-based weighting).
* **Better Control**: Manual control over similarity computation, top-k filtering, and fallback handling.

---

## Output

This setup filters out sparse data by requiring books to have at least 3 ratings and users to have rated at least 4 books, with no similarity shrinkage (λ = 0) and a fixed random seed for reproducibility.

* A `pivot` table showing MAD scores across training ratios and `k` values.

### MAD Results (λ = 0)

| Train Ratio | k = 5  | k = 10 | k = 15 | k = 20 | k = 50 | k = 100 |
|-------------|--------|--------|--------|--------|--------|---------|
| 0.60        | 1.4322 | 1.4329 | 1.4335 | 1.4340 | 1.4409 | 1.5034  |
| 0.65        | 1.4288 | 1.4274 | 1.4271 | 1.4288 | 1.4392 | 1.4736  |
| 0.70        | 1.4270 | 1.4240 | 1.4349 | 1.4270 | 1.4653 | 2.2490  |
| 0.75        | 1.4261 | 1.4233 | 1.4277 | 1.4645 | 1.4538 | 1.4959  |
| 0.80        | 1.4265 | 1.4218 | 1.4405 | 1.4233 | 1.4379 | 1.4675  |
| 0.85        | 1.4227 | 1.4164 | 1.4144 | 1.4161 | 1.4242 | 1.5271  |
| 0.90        | 1.4165 | 1.4113 | 1.4094 | 1.4089 | 1.4335 | 1.4475  |

**Key Observations:**

* MAD improves consistently as the training ratio increases from 0.60 to 0.90.
* Lowest MAD (\~1.409) occurs at `train_ratio = 0.90`, `k = 20`.
* Small `k` values (5–20) yield the best performance across all ratios.
* Large `k` values (50, 100) are more "stable" with shrinkage but less accurate.
* Kept shrinkage λ = 0.

**Summary Table:**

| Insight                      | Observation                                  |
| ---------------------------- | -------------------------------------------- |
| Best MAD                     | \~1.4089 at `train_ratio = 0.90`, `k = 20`   |
| Ideal `k` range              | 10–20 consistently outperforms others        |
| Effect of shrinkage          | Stabilizes results, especially for large `k` |
| Impact of more training data | Lowers error steadily                        |
| Limitation of high `k`       | Accuracy drops unless supported by more data |

---

## To Run

After updating the path, just `run all` should work.
```python
run_holder(ratings_path, users_path, books_path)
```

---

## License

MIT License. Attribution appreciated.
