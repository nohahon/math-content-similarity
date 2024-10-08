import sys
import json
import datasplits
import eval_metrics

def idlRecommendations(doc_id):
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = datasplits.load_csv_to_dataframe(data_)  # Load main dataset
    train_, test_, valid_ = datasplits.getData_de(dataFr)  # Split data into train, test, valid
    #print(test_['document_id'].dtype)
    val_ret = test_[test_['document_id'] == int(doc_id)]['citation_de'].values
    #print(val_ret)
    return val_ret

def read_results_file(file_path):
    # Open the JSON file in read mode
    with open(file_path, 'r') as json_file:
        # Load the content of the file into a Python dictionary
        data = json.load(json_file)
    genRecmnds, idealRecmnds = [], []
    for eackDoc in data.keys():
        idl_recmnds = idlRecommendations(eackDoc)
        idealRecmnds.append(idl_recmnds)
        gen_recmnds = [str(ea_) for ea_ in data[eackDoc][:10]]
        genRecmnds.append(gen_recmnds)
    p3, p5, r_, mrr_, ndcg_ = eval_metrics.main(idealRecmnds, genRecmnds)
    print(p3, p5, r_, mrr_, ndcg_)

# Example usage
if __name__ == "__main__":
    file_path = "ranked_documents.json"  # Replace with the path to your JSON file
    json_data = read_results_file(file_path)
