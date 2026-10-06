"""Step 1: load, clean and audit the Sephora skincare review data.

Every exclusion rule is logged to output/audit_log.csv so the analysis can be
re-run and checked. Usage: python src/01_clean.py data/ output/
"""
import glob
import os
import sys

import pandas as pd

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
log = []


def step(name, before, after, why):
    log.append({"step": name, "rows_before": before, "rows_after": after,
                "rows_removed": before - after, "reason": why})
    print(f"{name:<38} {before:>9,} -> {after:>9,}  ({why})")


# ---- reviews --------------------------------------------------------------
files = sorted(glob.glob(os.path.join(DATA, "reviews_*.csv")))
r = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
r = r.drop(columns=[c for c in r.columns if c.startswith("Unnamed")])
n0 = len(r)
step("Loaded review files", n0, n0, f"{len(files)} files")

n = len(r)
r = r.drop_duplicates()
step("Exact duplicate rows", n, len(r), "same review appeared twice")

n = len(r)
r = r.drop_duplicates(subset=["author_id", "product_id", "review_text"])
step("Same author, product and text", n, len(r), "resubmitted review")

n = len(r)
r = r[r["rating"].between(1, 5)]
step("Rating outside 1-5", n, len(r), "invalid rating")

n = len(r)
r = r[r["is_recommended"].notna()]
step("No recommend answer", n, len(r), "question not asked before 2018; missingness unrelated to rating")

r["submission_time"] = pd.to_datetime(r["submission_time"], errors="coerce")
r["year"] = r["submission_time"].dt.year
r["review_text"] = r["review_text"].fillna("")
r["text_len"] = r["review_text"].str.split().str.len()

# ---- products -------------------------------------------------------------
p = pd.read_csv(os.path.join(DATA, "product_info.csv"))
keep = ["product_id", "loves_count", "rating", "reviews", "price_usd", "limited_edition",
        "new", "online_only", "sephora_exclusive", "highlights", "primary_category",
        "secondary_category", "tertiary_category", "ingredients"]
p = p[keep].rename(columns={"rating": "product_rating", "reviews": "product_reviews",
                            "price_usd": "list_price_usd"})

n = len(r)
r = r.merge(p, on="product_id", how="inner")
step("Product not in product_info", n, len(r), "cannot attach category")

# check category scope
print("\nprimary_category of reviewed products:\n", r["primary_category"].value_counts())

# ---- profile completeness (kept, but flagged) -----------------------------
for c in ["skin_tone", "skin_type", "eye_color", "hair_color"]:
    print(f"{c:<10} missing: {r[c].isna().mean():.1%}")

r.reset_index(drop=True).to_pickle(os.path.join(OUT, "reviews_clean.pkl"))
pd.DataFrame(log).to_csv(os.path.join(OUT, "audit_log.csv"), index=False)
print(f"\nFinal: {len(r):,} reviews, {r['product_id'].nunique():,} products, "
      f"{r['author_id'].nunique():,} reviewers, {r['brand_name'].nunique():,} brands, "
      f"{r['submission_time'].min():%Y-%m} to {r['submission_time'].max():%Y-%m}")
