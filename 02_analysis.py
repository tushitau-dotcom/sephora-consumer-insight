"""Step 2: consumer insight analysis of Sephora skincare reviews.

Reads output/reviews_clean.pkl (from 01_clean.py) and writes every number used
in the insight deck to output/results.json, plus supporting CSV tables.
Usage: python src/02_analysis.py output/
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

OUT = sys.argv[1]
r = pd.read_pickle(os.path.join(OUT, "reviews_clean.pkl"))
r["t"] = r["review_text"].str.lower().str.replace("’", "'", regex=False)
R = {}

# ---------------------------------------------------------------- helpers
# A theme match only counts if no negation word sits just before it in the same
# sentence, so "didn't break me out" is not counted as a breakout complaint.
NEG = re.compile(r"(?:\bno\b|\bnot\b|n't|\bnever\b|\bwithout\b|\bzero\b|\bnor\b|"
                 r"free of|\bany\b)[^.!?]{0,20}$")


def flag(text, pattern):
    for m in re.finditer(pattern, text):
        if not NEG.search(text[:m.start()]):
            return True
    return False


TONE = {"porcelain": "Fair", "fair": "Fair", "fairLight": "Fair",
        "light": "Light", "lightMedium": "Light",
        "medium": "Medium", "mediumTan": "Medium", "tan": "Medium", "olive": "Medium",
        "deep": "Deep", "rich": "Deep", "dark": "Deep", "ebony": "Deep"}
r["tone"] = r["skin_tone"].map(TONE)

# ---------------------------------------------------------------- scope
R["scope"] = {
    "reviews": len(r), "products": int(r.product_id.nunique()),
    "brands": int(r.brand_name.nunique()), "reviewers": int(r.author_id.nunique()),
    "first": f"{r.submission_time.min():%b %Y}", "last": f"{r.submission_time.max():%b %Y}",
    "recommend_rate": round(r.is_recommended.mean() * 100, 1),
    "mean_rating": round(r.rating.mean(), 2),
}

# ---------------------------------------------------------------- stars vs recommend
x = r.groupby("rating").agg(n=("rating", "size"), rec=("is_recommended", "mean"))
x["share"] = x.n / x.n.sum()
R["stars"] = {int(k): {"share_pct": round(v.share * 100, 1), "rec_pct": round(v.rec * 100, 1)}
              for k, v in x.iterrows()}

# ---------------------------------------------------------------- 1. free-product reviews
FREE = (r"complimentary|received (?:this|it|the product)(?: product)? (?:for )?free|gifted|"
        r"got (?:this|it) (?:for )?free|free sample|influenster|in exchange for|"
        r"received this product (?:for free|free|complimentary|as a|from)|"
        r"free from (?:influenster|bzzagent|pinchme|the brand|sephora)")
r["free"] = r["t"].str.contains(FREE, regex=True)
f = r.groupby("free").agg(n=("rating", "size"), rec=("is_recommended", "mean"),
                          rat=("rating", "mean"))
R["free"] = {
    "share_pct": round(r.free.mean() * 100, 1),
    "n_free": int(f.loc[True, "n"]),
    "rec_free": round(f.loc[True, "rec"] * 100, 1), "rec_paid": round(f.loc[False, "rec"] * 100, 1),
    "rating_free": round(f.loc[True, "rat"], 2), "rating_paid": round(f.loc[False, "rat"], 2),
    "by_year": {int(k): round(v * 100, 1) for k, v in
                r[r.year >= 2018].groupby("year").free.mean().items()},
    "precision_check": "40 of 40 randomly sampled matches were genuine free-product reviews",
}

# same product, free vs paid reviewers (controls for product quality)
g = r.groupby(["product_id", "free"]).agg(n=("rating", "size"), rec=("is_recommended", "mean")).unstack()
g = g[(g[("n", False)] >= 50) & (g[("n", True)] >= 20)]
gap = (g[("rec", True)] - g[("rec", False)]) * 100
R["free"]["within_product"] = {"products": len(g), "median_gap_pp": round(gap.median(), 1),
                               "pct_products_free_higher": round((gap > 0).mean() * 100, 0)}

# what the shopper sees vs what paying customers say
a = r.groupby("product_id").agg(n=("rating", "size"), rec_all=("is_recommended", "mean"),
                                free_share=("free", "mean"))
o = r[~r.free].groupby("product_id").agg(n_paid=("rating", "size"), rec_paid=("is_recommended", "mean"))
a = a.join(o)
a = a[(a.n >= 100) & (a.n_paid >= 30)].copy()
a["inflation_pp"] = (a.rec_all - a.rec_paid) * 100
bands = pd.cut(a.free_share, [-0.01, 0.05, 0.2, 0.5, 1.0],
               labels=["Under 5%", "5-20%", "20-50%", "Over 50%"])
bt = a.groupby(bands, observed=False).agg(products=("n", "size"), inflation_pp=("inflation_pp", "mean"))
a["rank_all"] = a.rec_all.rank(ascending=False, method="first")
a["rank_paid"] = a.rec_paid.rank(ascending=False, method="first")
top = a[a.rank_all <= 100]
R["free"]["ranking"] = {
    "products": len(a),
    "bands": {str(k): {"products": int(v.products), "inflation_pp": round(v.inflation_pp, 1)}
              for k, v in bt.iterrows()},
    "top100_drop_out": int((top.rank_paid > 100).sum()),
    "top100_free_share": round(top.free_share.mean() * 100, 1),
    "all_free_share": round(a.free_share.mean() * 100, 1),
}
a.sort_values("inflation_pp", ascending=False).to_csv(os.path.join(OUT, "product_inflation.csv"))

# ---------------------------------------------------------------- 2. why people don't recommend
THEMES = {
    "Breakouts or irritation": r"broke me out|breaking me out|broken out|caused (?:me to )?break|breakouts?\b|"
                               r"\bburn(?:ed|s|ing|t)?\b|sting(?:s|ing|y)?\b|\brash\b|allergic|irritat\w*|hives",
    "No visible results": r"no difference|did nothing|does nothing|didn't do (?:anything|much)|"
                          r"doesn't do (?:anything|much)|no results|no improvement|no change|"
                          r"didn't see (?:any|a|much)|don't see (?:any|a|much)|didn't notice (?:any|a)|not see any",
    "Greasy, sticky or pilling": r"greasy|sticky|tacky|\bpill(?:s|ing|ed)?\b|oily residue|\bslimy\b",
    "Not worth the price": r"not worth|overpriced|too expensive|waste of money|save your money|for the price|"
                           r"for this price|price is too|so expensive|pricey",
    "Didn't live up to the hype": r"wanted to love|wanted to like|the hype|overrated|so many good reviews|"
                                  r"all the (?:good )?reviews|rave reviews",
    "Drying or tight": r"so dry|too drying|very drying|dried (?:out|my)|stripp\w+|feels tight|felt tight|dehydrat",
    "Strong scent": r"perfumey|strong (?:smell|scent|fragrance)|smells (?:bad|awful|weird|terrible|like chemicals|horrible)|"
                    r"overpowering|too much fragrance|scent is (?:too|very|really) strong|smell is (?:too|very|really) strong",
    "Formula changed": r"new formula|changed the formula|reformulat\w*|old formula",
    "Packaging failed": r"leak\w*|pump (?:broke|stopped|doesn't|does not|didn't|is broken|won't)|broken pump|"
                        r"unhygienic|wasteful packaging|packaging is (?:terrible|awful|bad|horrible|cheap)",
}
paid = r[~r.free]
nonrec = paid[paid.is_recommended == 0]
rec = paid[paid.is_recommended == 1].sample(150000, random_state=0)
rows, any_hit = [], pd.Series(False, index=nonrec.index)
for k, p in THEMES.items():
    hn = nonrec.t.map(lambda s: flag(s, p))
    hr = rec.t.map(lambda s: flag(s, p))
    any_hit |= hn
    rows.append({"theme": k, "pct_of_non_recommenders": round(hn.mean() * 100, 1),
                 "pct_of_recommenders": round(hr.mean() * 100, 1),
                 "times_more_likely": round(hn.mean() / hr.mean(), 1)})
th = pd.DataFrame(rows).sort_values("pct_of_non_recommenders", ascending=False)
th.to_csv(os.path.join(OUT, "complaint_themes.csv"), index=False)
R["themes"] = {"non_recommenders": len(nonrec), "pct_with_any_theme": round(any_hit.mean() * 100, 1),
               "table": th.to_dict(orient="records")}

# ---------------------------------------------------------------- 3. whose voice is heard
tone_share = r.tone.value_counts(normalize=True) * 100
R["tone"] = {"share_pct": {k: round(v, 1) for k, v in tone_share.items()},
             "reported_pct": round(r.tone.notna().mean() * 100, 1)}

d = r.dropna(subset=["tone"]).copy()
d["dev"] = d.is_recommended - d.groupby("product_id").is_recommended.transform("mean")
gaps = d.groupby("tone").dev.mean() * 100
R["tone"]["within_product_gap_pp"] = {k: round(v, 1) for k, v in gaps.items()}
d2 = r.dropna(subset=["skin_type"]).copy()
d2["dev"] = d2.is_recommended - d2.groupby("product_id").is_recommended.transform("mean")
R["skin_type"] = {"share_pct": {k: round(v * 100, 1) for k, v in d2.skin_type.value_counts(normalize=True).items()},
                  "within_product_gap_pp": {k: round(v * 100, 1) for k, v in d2.groupby("skin_type").dev.mean().items()}}
# largest category x tone gap with a reasonable deep-tone sample
ct = d.pivot_table(index="tertiary_category", columns="tone", values="dev", aggfunc="mean") * 100
cn = d.pivot_table(index="tertiary_category", columns="tone", values="dev", aggfunc="size")
ct = ct[cn["Deep"] >= 300]
R["tone"]["max_abs_category_gap_pp"] = round(float(ct.abs().max().max()), 1)

# sunscreen white cast, paid reviews
s = paid[paid.tertiary_category == "Face Sunscreen"].copy()
WC = r"white cast|whitecast|grey cast|gray cast|purple cast|ashy|\bcast\b"
s["mention"] = s.t.str.contains(WC, regex=True)
s["complain"] = s.t.map(lambda x: flag(x, WC))
w = s.groupby("tone").agg(n=("rating", "size"), mention=("mention", "mean"),
                          complain=("complain", "mean"), rec=("is_recommended", "mean"))
R["sunscreen"] = {k: {"reviews": int(v.n), "mention_pct": round(v.mention * 100, 1),
                      "complain_pct": round(v.complain * 100, 1), "rec_pct": round(v.rec * 100, 1)}
                  for k, v in w.iterrows()}

# ---------------------------------------------------------------- price
r["price_band"] = pd.cut(r.price_usd, [0, 25, 50, 75, 100, 10000],
                         labels=["Under $25", "$25-50", "$50-75", "$75-100", "$100+"])
pb = r.groupby("price_band", observed=True).agg(n=("rating", "size"), rec=("is_recommended", "mean"))
R["price"] = {str(k): {"reviews": int(v.n), "rec_pct": round(v.rec * 100, 1)} for k, v in pb.iterrows()}

with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(R, fh, indent=2, default=str)
print(json.dumps(R, indent=2, default=str))
