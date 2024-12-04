import os
import sys
import torch
import numpy as np
import pandas as pd
sys.path.append('../../tf_idf_algrthmn/')
import zbCitData_st
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

# Load the model and tokenizer
model_name = "dunzhang/stella_en_400M_v5"
vector_dim = 1024
vector_linear_directory = f"2_Dense_{vector_dim}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
vector_linear_dict = {
    k.replace("linear.", ""): v for k, v in
    torch.load(os.path.join("/beegfs/schubotz/.cache/huggingface/hub/models--dunzhang--stella_en_400M_v5/snapshots/24e2e1ffe95e95d807989938a5f3b8c18ee651f5", f"{vector_linear_directory}/pytorch_model.bin")).items()
}
vector_linear.load_state_dict(vector_linear_dict)
vector_linear.cuda()

# Load the checkpoint for the model
checkpoint_path = "chckpnts_stle_rand_ß24/checkpoint_special.pt"
checkpoint = torch.load(checkpoint_path)

# Load model weights from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode for inference
model.eval()

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Sample texts for comparison
#text1 = "This is the first document."
#text2 = "This is not the second document."

# zbRevCitData
data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
print("Model loaded")
dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
train_, test_, valid_ = zbCitData_st.split_dataframe(dataFr)# Split data into train, test, valid
main_data = zbCitData_st.getMainData()
missing_document_ids = test_[~test_['document_id'].isin(main_data['document_id'])]['document_id']
missing_data = pd.DataFrame({'document_id': missing_document_ids, 'text': 'No Abstract'})
main_data = pd.concat([main_data, missing_data], ignore_index=True)
batch_doc_ids = test_['document_id'][0:45]
batch_titles = main_data.set_index('document_id').loc[batch_doc_ids]['text'].values
print(type(batch_titles))
#sys.exit(0)

# Generate embeddings (no need to compute gradients during inference)
with torch.no_grad():
    input_data = tokenizer(batch_titles.tolist(), padding="longest", truncation=True, max_length=1024, return_tensors="pt")
    input_data = {k: v.cuda() for k, v in input_data.items()}
    attention_mask = input_data["attention_mask"]
    last_hidden_state = model(**input_data)[0]
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    query_vectors = normalize(vector_linear(query_vectors).cpu().numpy())
    similarities = query_vectors @ query_vectors.T

print(similarities)