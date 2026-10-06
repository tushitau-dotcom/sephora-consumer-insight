# Whose voice sells skincare?

A consumer insight study of 926,051 Sephora skincare reviews (2018 to 2023), asking how well the reviews a shopper reads reflect what paying customers actually think.

## Key findings

1. **Free-product reviews inflate scores.** 15% of reviews come from people who were given the product. They recommend it 92.7% of the time, against 82.4% for paying customers. Comparing reviewers of the same product, free reviewers recommend more in 80% of 733 products (median +5.6 points). 14 of the top 100 products by displayed recommend rate fall out of the top 100 when only paying customers are counted.
2. **Unmet expectations decide non-recommendation.** "No visible results" and "didn't live up to the hype" are 12 to 13 times more common in non-recommend reviews than in recommend reviews. Breakouts are the most frequent complaint but far less decisive (2x).
3. **Deep skin tones are satisfied but unheard.** My hypothesis that deep-tone reviewers recommend less was not supported: within the same product, every tone group is within 1 point of average. But deep-tone shoppers write only 2.9% of reviews. In face sunscreen they mention white cast in 46% of reviews (fair skin: 19%) and complain about it three times as often.

## Recommendations (UK beauty retailer)

- Show and rank on a paying-customer recommend rate. The DMCC Act 2024 (review provisions in force 6 April 2025) requires incentivised reviews to be clearly labelled.
- Set expectations on product pages ("results in X weeks", "best for") and check in two weeks after purchase.
- Point sampling at under-represented shoppers, add skin-tone review filters, and note cast performance on sunscreens.

## Method

| Step | Script | What it does |
|---|---|---|
| 1 | `src/01_clean.py` | Combines the five review files, removes 557 duplicates and the 167,803 pre-2018 reviews where the recommend question was not asked (missingness unrelated to rating), joins product data, writes `output/audit_log.csv` |
| 2 | `src/02_analysis.py` | Free-review detection, same-product comparisons, ranking impact, complaint themes, skin tone and type gaps, sunscreen white cast. Every figure in the deck is in `output/results.json` |
| 3 | `src/03_text_model.py` | TF-IDF + logistic regression on 300,000 paid reviews (hold-out AUC 0.98); its top phrases were used to choose the complaint themes |

**Free-review detection** uses disclosure phrases ("received this complimentary", "gifted", "Influenster", "in exchange for"). 40 of 40 randomly sampled matches were genuine. Undisclosed free reviews are missed, so estimates are conservative.

**Complaint themes** are negation-aware: a match preceded by "no", "not", "without" etc. in the same sentence is ignored, so "didn't break me out" is not counted as a breakout.

## Limitations

US data; free reviews detected only when disclosed; keyword themes are approximate; reviews only capture people who bought, so avoided products are invisible. Next step: a UK shopper survey testing the three open questions.

## Reproduce

```
pip install pandas scikit-learn
# download "Sephora Products and Skincare Reviews" from Kaggle into data/
python src/01_clean.py data output
python src/02_analysis.py output
python src/03_text_model.py output
```

Data: [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews), Kaggle. Raw data is not included in this repository.
