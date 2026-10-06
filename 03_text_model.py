"""Step 3: find which words and phrases predict "would not recommend".

Trains a TF-IDF + logistic regression model on 300,000 paid (non-free) reviews.
The top-weighted phrases were used to choose the complaint themes in
02_analysis.py. Writes output/text_model_terms.csv.
Usage: python src/03_text_model.py output/
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

OUT = sys.argv[1]
r = pd.read_pickle(os.path.join(OUT, "reviews_clean.pkl"))
t = r.review_text.str.lower()
FREE = (r"complimentary|received (?:this|it|the product)(?: product)? (?:for )?free|gifted|"
        r"got (?:this|it) (?:for )?free|free sample|influenster|in exchange for|"
        r"received this product (?:for free|free|complimentary|as a|from)|"
        r"free from (?:influenster|bzzagent|pinchme|the brand|sephora)")
paid = r[~t.str.contains(FREE, regex=True)].sample(300000, random_state=0)

vec = TfidfVectorizer(ngram_range=(1, 3), min_df=50, max_features=60000, sublinear_tf=True)
X = vec.fit_transform(paid.review_title.fillna("") + " . " + paid.review_text)
y = paid.is_recommended.astype(int)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
model = LogisticRegression(C=2, max_iter=2000, class_weight="balanced").fit(Xa, ya)
auc = roc_auc_score(yb, model.predict_proba(Xb)[:, 1])
print(f"Hold-out AUC: {auc:.3f}")

terms = pd.DataFrame({"term": vec.get_feature_names_out(), "weight": model.coef_[0]})
terms = terms.sort_values("weight")
terms.to_csv(os.path.join(OUT, "text_model_terms.csv"), index=False)
print("Most predictive of NOT recommending:", ", ".join(terms.term.head(40)))
print("Most predictive of recommending:", ", ".join(terms.term.tail(40)[::-1]))
