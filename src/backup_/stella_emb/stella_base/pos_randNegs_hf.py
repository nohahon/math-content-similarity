import random
import sys
import pandas as pd
from datasets import load_dataset, Dataset
sys.path.append('../../tf_idf_algrthmn/')
import zbCitData_st

def get_data_train(train_df, main_df):
    """return two lists first with samples and second with lables """
    records, labels = [], []
    setForNegDocs = set()
    train_citations = set(
        elem.strip()
        for citations in train_df['citation_de'].dropna()
        for elem in citations.split(';')
    )  # citations as indiv entries from train set
    train_docs = set(train_df['document_id'])
    setForNegDocs = setForNegDocs.union(train_docs).union(train_citations) # comb train docs and train cits
    # Convert 'document_id' column of main_df to a set for faster lookup
    main_document_ids = set(main_df['document_id'])
    setForNegDocs = main_document_ids - setForNegDocs # all doc ids minus train doc + cits to get neg samples (so that we avoid possibility of having doc id & cits as negs)
    trndocinth = main_document_ids.intersection(train_docs) #not necessary but just to check if all train docs are in main docs so that we don't have an error while getting text
    main_document_ids = set(str(each) for each in main_document_ids) # doc_ids are int but cits are string to converting doc_ids to string (did not do before bcz that might have not given intersection of doc_ids of train and doc_ids of main)
    # Find the intersection
    matching_elements = train_citations.intersection(main_document_ids) #train cits that are in main_docs Ids
    for index, row in train_df.iterrows():
        if row['document_id'] in trndocinth:
            seedStr = main_df.loc[main_df['document_id'] == row['document_id'], 'text'].iloc[0]
            cits_ = row['citation_de'].split(';')
            neg_cnt = 0
            for ech_cit in cits_:
                if ech_cit in matching_elements:
                    neg_cnt += 1
                    idl_rcmnd_str = main_df.loc[main_df['document_id'] == int(ech_cit), 'text'].iloc[0]
                    records.append(seedStr+" "+idl_rcmnd_str)
                    labels.append(1.0)
            random_elements = random.sample(setForNegDocs, neg_cnt)
            for eachRnd in random_elements:
                neg_rcmnds = main_df.loc[main_df['document_id'] == eachRnd, 'text'].iloc[0]
                records.append(seedStr+" "+neg_rcmnds)
                labels.append(0.0)
    print(len(records), len(labels))
    return records, labels


def dataon_hf():
    """ Put train test and validation data on huggingface """
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
    main_data = zbCitData_st.getMainData()
    train_, test_, valid_ = zbCitData_st.getData_de(dataFr)  # Split data into train, test, valid
    records, labels = [], []
    print(list(train_), list(test_), list(valid_), list(main_data))

    list_trainsamp, train_labl = get_data_train(train_, main_data)
    data_dict = {"text": list_trainsamp, "label": train_labl}
    ds_ = Dataset.from_dict(data_dict)
    ds_.push_to_hub("AnkitSatpute/zbMath_contra_rand", split="train")

    list_trainsamp, train_labl = get_data_train(test_, main_data)
    data_dict = {"text": list_trainsamp, "label": train_labl}
    ds_ = Dataset.from_dict(data_dict)
    ds_.push_to_hub("AnkitSatpute/zbMath_contra_rand", split="test")

    list_trainsamp, train_labl = get_data(valid_, main_data)
    data_dict = {"text": list_trainsamp, "label": train_labl}
    ds_ = Dataset.from_dict(data_dict)
    ds_.push_to_hub("AnkitSatpute/zbMath_contra_rand", split="validation")

#dataon_hf()
