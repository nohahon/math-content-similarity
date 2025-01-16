import sys
import pandas as pd

def load_main_data():
    main_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_complete_cleaned.csv')
    co_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_extkw.csv')
    main_data_ = pd.merge(main_data_,co_data_, on='document_id')
    main_data_ = main_data_.fillna('')
    print(main_data_.columns)
    return main_data_

load_main_data()
