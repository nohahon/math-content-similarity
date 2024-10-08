import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_csv_to_dataframe(file_path):
    """ Function to load CSV into a pandas dataframe """
    df = pd.read_csv(file_path)
    df = df[df['reviewer'] != 0] # Excluding docs from reviewer 0 since we dont know if they are legit
    return df

def count_and_check_unique_ids(df):
    """ Function to count rows by document_id and check uniqueness """
    row_count = df['document_id'].count()
    unique_ids = df['document_id'].is_unique
    return row_count, unique_ids

def analyze_citation_de(df):
    """Analyze citation column to see pairs"""
    # Split citation_de column by "; " and calculate the number of elements per row
    citation_counts = df['citation_de'].str.split("; ").apply(len)
    # Calculate min, avg, and max citation counts
    min_citations = citation_counts.min()
    avg_citations = citation_counts.mean()
    max_citations = citation_counts.max()
    all_citations = citation_counts.sum()
    # Create unique pairs of document_id and each citation in citation_de
    unique_pairs = df[['document_id', 'citation_de']].copy()
    unique_pairs['citation_de'] = unique_pairs['citation_de'].str.split("; ")
    unique_pairs = unique_pairs.explode('citation_de').drop_duplicates()
    unique_pairs_count = unique_pairs.shape[0]
    return min_citations, avg_citations, max_citations,all_citations, unique_pairs_count

def analyzeReviewers(df):
    # Count unique reviewers
    df_filtered = df[df['reviewer'] != 0]
    unique_reviewers = df['reviewer'].nunique()
    # Group by reviewer and count unique document_id per reviewer
    reviewer_doc_counts = df.groupby('reviewer')['document_id'].nunique()
    threshold = 2
    # printing reviewer's IDs if they have document greater than certain threashol
    reviewers_above_threshold = reviewer_doc_counts[reviewer_doc_counts < threshold]
    if not reviewers_above_threshold.empty:
        print(f"Reviewer IDs with more than {threshold} documents:")
        print(reviewers_above_threshold)
    else:
        print(f"No reviewers with more than {threshold} documents found.")
    # Calculate min, max, and average document count per reviewer
    min_docs = reviewer_doc_counts.min()
    max_docs = reviewer_doc_counts.max()
    avg_docs = reviewer_doc_counts.mean()
    # Generate frequency distribution chart
    freq_distribution = reviewer_doc_counts.value_counts().sort_index()
    # Plotting the distribution
    #plt.figure(figsize=(10, 6))
    #plt.bar(freq_distribution.index, freq_distribution.values)
    #plt.xlabel('Number of Documents per Reviewer')
    #plt.ylabel('Number of Reviewers')
    #plt.title('Distribution of Documents per Reviewer')
    #plt.xticks(freq_distribution.index)
    #plt.grid(axis='y')
    #plt.savefig('frequency_reviewer.png')
    return unique_reviewers, min_docs, max_docs, avg_docs

def split_list(doc_ids, train_ratio, test_ratio, val_ratio):
    total_count = len(doc_ids)
    # Compute split sizes
    train_size = math.floor(total_count * train_ratio)
    test_size = math.floor(total_count * test_ratio)
    val_size = math.floor(total_count * val_ratio)
    # If the sizes don't perfectly sum up to total_count, add remaining samples to train
    remainder = total_count - (train_size + test_size + val_size)
    train_size += remainder  # Add remaining samples to train
    train_split = doc_ids[:train_size]
    test_split = doc_ids[train_size:train_size + test_size]
    val_split = doc_ids[train_size + test_size:]
    return train_split, test_split, val_split

def split_dataframe(df, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):
    citation_length_dict = {}
    # Populate the dictionary
    for index, row in df.iterrows():
        doc_id = row['document_id']
        citations = row['citation_de'].split(';')
        length = len(citations)
        if length not in citation_length_dict:
            citation_length_dict[length] = []
        citation_length_dict[length].append(doc_id)
    train_ids, test_ids, validation_ids = [], [], []
    for length, doc_ids in citation_length_dict.items():
        train_split, test_split, val_split = split_list(doc_ids, train_ratio, test_ratio, val_ratio)
        train_ids.extend(train_split)
        test_ids.extend(test_split)
        validation_ids.extend(val_split)
    # Filter the original dataframe to create the final train, test, validation sets
    train_ = df[df['document_id'].isin(train_ids)][['document_id', 'citation_de']]
    test_ = df[df['document_id'].isin(test_ids)][['document_id', 'citation_de']]
    validation_ = df[df['document_id'].isin(validation_ids)][['document_id', 'citation_de']]
    return train_, test_, validation_

def getData_de(df):
    """ Dont use this: Incosistent splits """
    """Analyze citation column and split unique pairs into train, test, and validation sets."""
    # Split citation_de column by "; " and calculate the number of elements per row
    citation_counts = df['citation_de'].str.split("; ").apply(len)
    # Create unique pairs of document_id and each citation in citation_de
    unique_pairs = df[['document_id', 'citation_de']].copy()
    unique_pairs['citation_de'] = unique_pairs['citation_de'].str.split("; ")
    unique_pairs = unique_pairs.explode('citation_de').drop_duplicates()
    # Group by document_id and count citation_de per document_id
    citation_count_df = unique_pairs.groupby('document_id').size().reset_index(name='citation_count')
    # Merge citation_count back into unique_pairs
    unique_pairs = pd.merge(unique_pairs, citation_count_df, on='document_id')
    # Create train, test, and validation sets by citation_count groups
    train_list = []
    test_list = []
    validation_list = []
    # Group by citation count
    for citation_count, group in unique_pairs.groupby('citation_count'):
        # Split the group into train (60%), test (20%), and validation (20%)
        if len(group) > 1:
            train, temp = train_test_split(group, test_size=0.4, random_state=42)  # 40% goes to test + validation
            test, validation = train_test_split(temp, test_size=0.5, random_state=42)  # Split 20%/20% for test/validation
        else:
            # If only one pair in the group, directly assign it to validation
            validation = group
            train = pd.DataFrame(columns=group.columns)
            test = pd.DataFrame(columns=group.columns)
        # Append the results to the respective lists
        train_list.append(train)
        test_list.append(test)
        validation_list.append(validation)
    # Concatenate all train, test, and validation sets
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    validation_df = pd.concat(validation_list, ignore_index=True)
    return train_df, test_df, validation_df

def getMainData():
    data_absr = "/beegfs/schubotz/noah/arxMLiv/zbmath_abstracts.csv"
    data_titles = "/beegfs/schubotz/ankit/data/zbMATH_titles.csv"
    df1 = pd.read_csv(data_titles)
    df1 = df1.dropna(subset=['title'])
    df2 = pd.read_csv(data_absr)
    df2 = df2[['document_id', 'text']]
    df2 = df2.dropna(subset=['text'])
    merged_df = pd.merge(df1, df2, on='document_id')
    return merged_df

def main():
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = load_csv_to_dataframe(data_)
    print("Total docs & are the documnet IDs unique? ",count_and_check_unique_ids(dataFr))
    print("Citations min, avg , and max, count of all citations, uniq doc to citation pairs", analyze_citation_de(dataFr))
    print(analyzeReviewers(dataFr))

#main()