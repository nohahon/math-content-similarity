# %%
import pandas as pd
import polars as pl
from tqdm import tqdm
import numpy as np
import math

tqdm.pandas()
np.random.seed(42)

# %%
header = ["seed"] + ["rec" + str(i) for i in range(14)]
recs = pd.read_csv("../data/zbmath_golden.csv", names=header, index_col=0)
train_ids = pd.read_csv("../data/train_ids.csv")

full = pd.read_csv("../data/zbmath_open.csv")
full_abstracts = pd.read_csv("../data/zbmath_open_abstracts.csv")


# %%
merged = full.merge(
    full_abstracts,
    left_on="de",
    right_on="document_id",
    how="outer",
)
merged = merged[(~merged["text_x"].isna()) | (~merged["text_y"].isna())]


def aggregate_ids(id1, id2):
    if math.isnan(id1) and math.isnan(id2):
        return math.nan
    if math.isnan(id1) and not math.isnan(id2):
        return int(id2)
    if not math.isnan(id1) and math.isnan(id2):
        return int(id1)
    if not math.isnan(id1) and not math.isnan(id2):
        if int(id1) != int(id2):
            return math.nan
        else:
            return int(id1)


def aggregate_texts(text_x, text_y):
    text_x = str(text_x)
    text_y = str(text_y)

    is_useless = lambda x: len(x) < 100

    if is_useless(text_x) and is_useless(text_y):
        return math.nan
    if is_useless(text_y):
        return text_x
    else:
        return text_y


def aggregate_keyword(text_x, text_y):
    isna_ = lambda x: x != x

    if isna_(text_x) and isna_(text_y):
        return math.nan
    if isna_(text_x):
        keyword_lst = text_y.split("; ")
        if len(keyword_lst) < 2:
            return math.nan
        else:
            return keyword_lst
    else:
        return text_x


def aggregate_msc(text_x, text_y):
    isna_ = lambda x: x != x

    if isna_(text_x) and isna_(text_y):
        return math.nan
    if isna_(text_x):
        return text_y.split(" ")
    else:
        return text_x


# %%
merged["id"] = merged.progress_apply(
    lambda x: aggregate_ids(x.de, x.document_id),
    axis=1,
)
merged["text"] = merged.progress_apply(
    lambda x: aggregate_texts(x.text_x, x.text_y),
    axis=1,
)
merged["keywords"] = merged.progress_apply(
    lambda x: aggregate_keyword(x.keyword_x, x.keyword_y),
    axis=1,
)
merged["MSC"] = merged.progress_apply(
    lambda x: aggregate_msc(x.msc, x.classification),
    axis=1,
)

# %%
merged = merged[["id", "text", "title", "keywords", "MSC"]]
merged = merged[~merged.text.isna()]
merged["keywords"] = merged["keywords"].astype(str)
merged["MSC"] = merged["MSC"].astype(str)

# %%
pl_f = pl.from_pandas(merged)
pl_f.write_parquet("../data/zbmath_open_clean.parquet")
