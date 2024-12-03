import random
import sys
import pandas as pd
from datasets import load_dataset, Dataset
sys.path.append('../../tf_idf_algrthmn/')
import zbCitData_st
from collections import defaultdict
from itertools import combinations

def get_data_train(train_df, main_df):
    """return two lists first with samples and second with lables """
    records, labels = [], []
    train_citations = set(
        elem.strip()
        for citations in train_df['citation_de'].dropna()
        for elem in citations.split(';')
    )
    main_document_ids = set(main_df['document_id'])
    dictSamps = defaultdict(lambda: 0)
    pos_docs_ = main_document_ids.intersection(train_citations)
    print("Lenth of all citations is: ", len(pos_docs_))
    train_docs = set(train_df['document_id'])
    setForNegDocs = train_docs.union(train_citations)
    setForNegDocs = main_document_ids - setForNegDocs # ensuring there are no doc ids from training set
    setForNegDocs = list(setForNegDocs)
    doc_text_dict = dict(zip(main_df['document_id'].astype(str), main_df['text']))
    for index, row in train_df.iterrows():
        docs_pos = list()
        docs_pos.append(row['document_id'])
        docs_pos += [eles_ for eles_ in row['citation_de'].split(';') if eles_ in pos_docs_]
        dictSamps[len([eles_ for eles_ in row['citation_de'].split(';') if eles_ in pos_docs_])] += 1
        pos_combs = list(combinations(docs_pos, 2))
        if len(pos_combs) > 0:
            random_elements = random.sample(setForNegDocs, len(pos_combs))
            random_elements.append(row['document_id'])
            neg_combs = list(combinations(random_elements ,2))
            for eachComb in pos_combs + neg_combs:
                doc1_str = doc_text_dict.get(str(eachComb[0]), "")
                doc2_str = doc_text_dict.get(str(eachComb[1]), "")
    print(dictSamps)
    print(len(records), len(labels))
    return records, labels

def dataon_hf():
    """ Put train test and validation data on huggingface """
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
    main_data = zbCitData_st.getMainData()
    train_, test_, valid_ = zbCitData_st.split_dataframe(dataFr)  # Split data into train, test, valid
    records, labels = [], []
    main_data["document_id"] = pd.to_numeric(main_data["document_id"], errors="coerce")
    main_data["document_id"] = main_data["document_id"].fillna(0).astype("string")
    print(list(train_),train_.shape[0], list(test_),test_.shape[0], list(valid_),valid_.shape[0] , list(main_data))

    list_trainsamp, train_labl = get_data_train(train_, main_data)
    data_dict = {"text": list_trainsamp, "label": train_labl}
    ds_ = Dataset.from_dict(data_dict)
    ds_.push_to_hub("AnkitSatpute/zbMath_contra_diff", split="train")

    list_trainsamp, train_labl = get_data_train(test_, main_data)
    data_dict = {"text": list_trainsamp, "label": train_labl}
    ds_ = Dataset.from_dict(data_dict)
    ds_.push_to_hub("AnkitSatpute/zbMath_contra_diff", split="test")

    list_trainsamp, train_labl = get_data_train(valid_, main_data)
    data_dict = {"text": list_trainsamp, "label": train_labl}
    ds_ = Dataset.from_dict(data_dict)
    ds_.push_to_hub("AnkitSatpute/zbMath_contra_diff", split="validation")

dataon_hf()