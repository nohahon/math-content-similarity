import sys
import pickle
import faiss
import json
import numpy as np
import pandas as pd
sys.path.append("../")
import datasplits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to compute TF-IDF and return top 1000 ranked document IDs by similarity to the title of docu_id
def rank_documents_by_similarity(main_data, docu_id):
    # Extract the title for the given docu_id
    try:
        query_title = main_data[main_data['document_id'] == docu_id]['title'].values[0]
    except:
        return [docu_id]
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=50000)  # Adjust max_features if needed for performance
    # Fit the vectorizer on the titles in the main data and transform them into TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(main_data['title'])
    # Transform the query title into the TF-IDF vector
    query_tfidf = vectorizer.transform([query_title])
    # Compute cosine similarity between the query title and all titles in main_data
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    # Rank document indices based on cosine similarity
    ranked_indices = cosine_similarities.argsort()[::-1][:1000]  # Top 1000 similar documents
    # Retrieve the top 1000 'document_id's based on similarity ranking
    ranked_document_ids = main_data['document_id'].iloc[ranked_indices].tolist()
    return ranked_document_ids

def main_tfidf():
    # Load your data (adjust according to your actual loading method)
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = datasplits.load_csv_to_dataframe(data_)  # Load main dataset
    train_, test_, valid_ = datasplits.getData_de(dataFr)  # Split data into train, test, valid
    main_data = datasplits.getMainData()  # Load main data containing 'document_id', 'title', 'text'
    results = {}  # Dictionary to store the results in {document_id: [top 1000 document_ids]}
    # Loop through each document_id in the test_ dataframe
    for batch_ in range(0, len(test_['document_id']), 100):
        print(batch_)
        for docu_id in test_['document_id'][batch_:batch_+100]:
            #print(len(test_['document_id'][batch_:batch_+100]))
            # Get the top 1000 ranked documents by similarity
            ranked_docs = rank_documents_by_similarity(main_data, docu_id)
            # Store the result in the dictionary with document_id as the key
            results[docu_id] = ranked_docs  # Convert to list to store in JSON format
            #sys.exit(0)
        # Save the results dictionary to a JSON file
        results = {int(doc_id): [int(idx) for idx in ranked_doc_ids] for doc_id, ranked_doc_ids in results.items()}
        with open('data_/ranked_documents_'+str(batch_)+'.json', 'w') as json_file:
            json.dump(results, json_file)
        results = {}
        print("Top 1000 ranked documents saved to 'ranked_documents.json'")

def normalize_L2(mat):
    """To nromalize dense matrx """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / norms

def createIndex(main_data_):
    """ Creating FAISS vector index """
    vectorizer = TfidfVectorizer(max_features=1000)
    pickle.dump(vectorizer, open('data/tfidf_vectorizer.pkl', 'wb'))
    tfidf_matrix = vectorizer.fit_transform(main_data_['title'])
    index = faiss.IndexFlatIP(tfidf_matrix.shape[1])  # Initialize with 5000 dimensions
    batch_size = 10000
    for start_idx in range(0, tfidf_matrix.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, tfidf_matrix.shape[0])
        tfidf_matrix_batch = tfidf_matrix[start_idx:end_idx].toarray().astype('float32')#get sparsetodens
        tfidf_matrix_batch = normalize_L2(tfidf_matrix_batch) #normalize
        index.add(tfidf_matrix_batch)
    faiss.write_index(index, "data/faiss_tfidf.index")

def main_faiss():
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = datasplits.load_csv_to_dataframe(data_)  # Load main dataset
    train_, test_, valid_ = datasplits.split_dataframe(dataFr)
    #train_, test_, valid_ = datasplits.getData_de(dataFr)  # Split data into train, test, valid
    main_data = datasplits.getMainData()
    missing_document_ids = test_[~test_['document_id'].isin(main_data['document_id'])]['document_id']
    missing_data = pd.DataFrame({'document_id': missing_document_ids, 'title': 'No title'})
    main_data = pd.concat([main_data, missing_data], ignore_index=True)
    #createIndex(main_data)
    # Loop through each document_id in the test_ dataframe
    index = faiss.read_index("data/faiss_tfidf.index")
    vectorizer = pickle.load(open('data/tfidf_vectorizer.pkl', 'rb'))
    vectorizer.fit(main_data['title'])
    batch_size = 5000
    for batch_start in range(0, len(test_['document_id']), batch_size):
        results = {}
        batch_doc_ids = test_['document_id'][batch_start:batch_start + batch_size]
        # Extract the titles for the batch and vectorize them
        batch_titles = main_data.set_index('document_id').loc[batch_doc_ids]['title'].values
        query_vectors = vectorizer.transform(batch_titles).toarray().astype('float32')
        faiss.normalize_L2(query_vectors)
        _, ranked_indices = index.search(query_vectors, 1000)
        for i, docu_id in enumerate(batch_doc_ids):
            ranked_doc_ids = [main_data['document_id'].iloc[idx_] for idx_ in ranked_indices[i]]
            results[docu_id] = ranked_doc_ids
        #print(results)
        results = {int(doc_id): [int(idx) for idx in ranked_doc_ids] for doc_id, ranked_doc_ids in results.items()}
        #print(results)
        with open(f'data/base_/title/tfidf_base_documents_{batch_start}.json', 'w') as json_file:
            json.dump(results, json_file)
        #sys.exit(0)

if __name__ == "__main__":
    #main_tfidf()
    main_faiss()