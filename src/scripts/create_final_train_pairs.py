# %%
import pandas as pd
from tqdm import tqdm
import numpy as np

tqdm.pandas()

np.random.seed(42)

# %%
conts = pd.read_csv("../data/zbmath_golden_contents.csv")
header = ["seed"] + ["rec" + str(i) for i in range(14)]
recs = pd.read_csv("../data/zbmath_golden.csv", names=header, index_col=0)
train_ids = pd.read_csv("../data/train_ids.csv")

full = pd.read_parquet("../data/zbmath_open_clean.parquet")

# %%
conts = conts.rename(
    columns={"Abstract/Review/Summarry": "abstract_review_summarry"},
)
conts["zbMATH_ID"] = conts["zbMATH_ID"].astype(int)
conts = conts.set_index("zbMATH_ID")

# %%
train_ids = train_ids["seeID"].unique()
test_ids = np.setdiff1d(np.array(recs.index), train_ids)
pool_ids = np.array(full.index)
all_labeled_ids = np.array(recs.index)

# %%
train_recs = recs.loc[train_ids]
test_recs = recs.loc[test_ids]


# %%
def create_pairs(df, recs_ids, full_ids):
    rowlist = []
    for seed, row in tqdm(df.iterrows()):
        rec_list = row[~row.isna()].astype(int).to_list()
        for rec_id in rec_list:
            # create positive
            rowlist.append({"seed": seed, "rec": rec_id, "label": 1})
            # sample negative
            random_negative = np.random.choice(full_ids)
            while random_negative in recs_ids:
                random_negative = np.random.choice(full_ids)
            rowlist.append({"seed": seed, "rec": random_negative, "label": 0})
    return pd.DataFrame(rowlist)


# %%
train_dataset = create_pairs(train_recs, all_labeled_ids, full_ids=pool_ids)
test_dataset = create_pairs(test_recs, all_labeled_ids, full_ids=pool_ids)


# %%
def get_contents(df, lookup_positive, lookup_negative):
    rowlist = []
    for idx, row in df.iterrows():
        if row["seed"] not in lookup_positive.index:
            continue
        anchor = lookup_positive.loc[row["seed"]].text
        # positive recommendations
        if row["label"] == 1:
            # skip the ones that are not present in contents
            if row["rec"] not in lookup_positive.index:
                continue

            rec = lookup_positive.loc[row["rec"]].text

        # negative ones. Sampling from whole zbmath corpus
        else:
            rec = lookup_negative.loc[row["rec"]].text

        rowlist.append(
            {
                "seed": row["seed"],
                "anchor": anchor,
                "rec": rec,
                "label": row["label"],
            },
        )
    return (
        pd.DataFrame(rowlist)
        .set_index("seed")
        .sample(frac=1.0, random_state=42)
    )


# %%
final_train_dataset = get_contents(train_dataset, conts, full)
final_test_dataset = get_contents(test_dataset, conts, full)

# %%
dev_ids = np.array([1745734, 1031529, 1275776, 1269765])  # set by Ankit
train_ids = np.setdiff1d(final_train_dataset.index.unique(), dev_ids)

# %%
final_dev_dataset = final_train_dataset.loc[dev_ids].sample(
    frac=1,
    random_state=42,
)
final_train_dataset = final_train_dataset.loc[train_ids].sample(
    frac=1,
    random_state=42,
)

# %%
final_train_dataset.to_csv("../data/final/final_train_dataset.csv")
final_dev_dataset.to_csv("../data/final/final_dev_dataset.csv")
final_test_dataset.to_csv("../data/final/final_test_dataset.csv")
