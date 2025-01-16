import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.append("/beegfs/schubotz/ankit/code/zbReviewCit/tf_idf_algrthmn")
import zbCitData_st

def load_main_data():
    main_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_complete_cleaned.csv')
    co_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_extkw.csv')
    main_data_ = pd.merge(main_data_,co_data_, on='document_id')
    main_data_ = main_data_.fillna('')
    return main_data_

def filter_by_last_ref(myFirst, refList):
    """Filters myFirst list to include only elements up to the last found document in refList."""
    # Find the indices in myFirst whose first element matches any element in refList
    last_index = -1  # Default index if no matches are found
    for i, item in enumerate(myFirst):
        if item[0] in refList:
            last_index = i
    return myFirst[:last_index + 1] if last_index != -1 else []

def getInitialRankedRes(seed_, idl_recmnds):
    model_dir = "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/"
    abstr_fet = "text_vs_text/scores/test/"
    getallfls = os.listdir(model_dir+abstr_fet)
    allabsdicts = []
    for i,eachF in enumerate(getallfls):
        with open(model_dir+abstr_fet+eachF, 'r') as json_file:
            abstr_ = json.load(json_file)
        allabsdicts.append(abstr_)
    abstr_ = {key: value for d in allabsdicts for key, value in d.items()}
    abs_ = filter_by_last_ref(abstr_[seed_], idl_recmnds) 
    title_fet = "title_vs_title/scores/test/"
    getallfls = os.listdir(model_dir+title_fet)
    alltitdicts = []
    for i,eachF in enumerate(getallfls):
        with open(model_dir+title_fet+eachF, 'r') as json_file:
            title_ = json.load(json_file)
        alltitdicts.append(title_)
    title_ = {key: value for d in alltitdicts for key, value in d.items()}
    tit_ = filter_by_last_ref(title_[seed_], idl_recmnds)
    kwrd_fet = "keyword_vs_keyword/scores/test/"
    getallfls = os.listdir(model_dir+kwrd_fet)
    allkwrddicts = []
    for i,eachF in enumerate(getallfls):
        with open(model_dir+kwrd_fet+eachF, 'r') as json_file:
            kwrd_ = json.load(json_file)
        allkwrddicts.append(kwrd_)
    kwrd_ = {key: value for d in allkwrddicts for key, value in d.items()}
    kwr_ = filter_by_last_ref(kwrd_[seed_], idl_recmnds)
    msc_fet = "mscs_in_text_vs_mscs_in_text/scores/test/"
    getallfls = os.listdir(model_dir+msc_fet)
    allmscdicts = []
    for i,eachF in enumerate(getallfls):
        with open(model_dir+msc_fet+eachF, 'r') as json_file:
            msc_ = json.load(json_file)
        allmscdicts.append(msc_)
    msc_ = {key: value for d in allmscdicts for key, value in d.items()}
    ms_ = filter_by_last_ref(msc_[seed_], idl_recmnds)
    return abs_, tit_, kwr_, ms_
    
def get_data_r():
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
    train_, test_, valid_ = zbCitData_st.split_dataframe(dataFr)
    print(test_.columns)
    model_dir = "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/"
    abstr_fet = "text_vs_text/scores/test/"
    getallfls = os.listdir(model_dir+abstr_fet)
    for i,eachF in enumerate(getallfls):
        with open(model_dir+abstr_fet+eachF, 'r') as json_file:
            abstr_ = json.load(json_file)
    count_lsts = 0
    records, seedrecPrs = [], []
    for eachK in abstr_.keys():
        citations_ = test_.loc[test_['document_id'] == float(eachK), 'citation_de'].values[0]
        citations_ = [int(cit_) for cit_ in citations_.split('; ')]
        abstr_, title_, kwrd_, msc_ = getInitialRankedRes(eachK, citations_)
        if len(abstr_) > 0:
            count_lsts += 1
        for id_,eachF in enumerate([abstr_, title_, kwrd_, msc_]):
            for eachEle in eachF:
                if eachEle[0] in citations_:
                    records.append([eachEle[1],id_,1.0])
                    seedrecPrs.append([eachK,eachEle[0]])
                else:
                    records.append([eachEle[1],id_,0.0])
                    seedrecPrs.append([eachK,eachEle[0]])
    with open("randomFOrest_data.pkl", "wb") as wpf:
        pickle.dump(seedrecPrs, wpf)
    print(count_lsts)
    return records

get_data_r()
