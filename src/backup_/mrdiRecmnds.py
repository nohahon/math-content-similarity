import csv
import sys
import pandas as pd

def load_main_data():
    main_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_complete_cleaned.csv')
    co_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_extkw.csv')
    main_data_ = pd.merge(main_data_,co_data_, on='document_id')
    main_data_ = main_data_.fillna('')
    print(main_data_.columns, main_data_.shape)
    return main_data_

def classification_similarity(codes_a, codes_b):
    """Returns the size of the intersection of two sets of classification codes."""
    return len(codes_a.intersection(codes_b))

def get_top5_similar(df):
    """For each row in df, return a list of the top 5 most similar document_ids 
    based on intersection of classification codes."""
    df_results = df.copy()
    # We'll store the top 5 similar doc_ids in a new column
    top5_col = []
    for idx, row in df.iterrows():
        current_codes = row['classification_codes']
        current_id = row['document_id']
        # Calculate similarity of the current row to all other rows
        similarities = []
        for idx2, row2 in df.iterrows():
            if idx2 == idx:
                continue  # skip comparing to itself
            sim_score = classification_similarity(current_codes, row2['classification_codes'])
            similarities.append((row2['document_id'], sim_score))

        # Sort by sim_score descending, then by document_id ascending (just in case of ties)
        similarities.sort(key=lambda x: (-x[1], x[0]))

        # Take top 5
        top5 = similarities[:5]

        # Extract just document_ids or keep the score as well
        top5_ids = [doc_id for doc_id, score in top5]
        top5_col.append(top5_ids)

    df_results['top5_similar'] = top5_col
    return df_results

get_data = load_main_data()
#print(get_data.iloc[20])
df_clean = get_data[
    get_data['classification'].notna() &
    (get_data['classification'].str.strip() != '')
]
#print(df_clean.iloc[20])

df_clean['classification_codes'] = df_clean['classification'].apply(
    lambda x: set(x.split())  # split on whitespace -> convert to set
)
df_ranked = get_top5_similar(df_clean)
#print("DataFrame with top5 similar:\n", df_ranked[['document_id', 'classification', 'top5_similar']])

# Select the relevant columns
df_to_save = df_ranked[['document_id', 'top5_similar']]

# Save to CSV
output_file = 'top5_similar_results.csv'
df_to_save.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

