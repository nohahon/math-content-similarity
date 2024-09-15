import sys
import jsonlines
import pickle
import createFTdata
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
sys.path.append('../../../longformer/')
import longformer_embed

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" #would be changed to use FT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Put the model in evaluation mode

def saveRepsonses():
    initPropt = createFTdata.getPrompt()
    endPrompt = " Please answer only in either Yes or No whether the candidate document can act as recommendation for the seed docuemnt. Do not explain your answer."
    loc_testdata = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/test_posandnegPairs.pkl"
    seedToPotrec = longformer_embed.getseedpotrec(loc_testdata)
    feat_ = longformer_embed.load_features()
    batchSize = 15
    msgbatch = []
    potrec = []
    with jsonlines.open('data/orign_resp_bin.jsonl', mode='w') as writer:
        for eachSeed in seedToPotrec.keys():
            seedStr = "Seed document: ["+createFTdata.getdocString(eachSeed, feat_)+"]."
            for eachPotRec in seedToPotrec[eachSeed]:
                if getdocString(eachPotRec, feat_) != "":
                    potrecstr = "Candidate document: ["+createFTdata.getdocString(idlrec, feat_)+" ]."
                    msgbatch.append(initPropt+seedStr+potrecstr+endPrompt)
                    potrec.append(eachPotRec)
                    if len(msgbatch) == batchSize:
                        results_ = getLogitLLaMa(msgbatch)
                        for id_,eachResult in enumerate(results_):
                            resultsdict = {"seed": None, "pot_rec": None, "response":None, "probab":None}
                            resultsdict["seed"] = eachSeed
                            resultsdict["pot_rec"] = potrec[id_]
                            resultsdict["response"] = eachResult[0] 
                            resultsdict["probab"] = eachResult[1]
                            writer.write(resultsdict)
                        writer.close()
                        writer = jsonlines.open('data/orign_resp_bin.jsonl', mode='a')
                        msgbatch = []
                        potrec = []
        if msgbatch:
            results_ = getLogitLLaMa(msgbatch)
            for id_,eachResult in enumerate(results_):
                resultsdict = {"seed": None, "pot_rec": None, "response":None, "probab":None}
                resultsdict["seed"] = eachSeed
                resultsdict["pot_rec"] = potrec[id_]
                resultsdict["response"] = eachResult[0] #reponse[0]['generated_text'][1]
                resultsdict["probab"] = eachResult[1]
                writer.write(resultsdict)
                writer.close()

def getLogitLLaMa(msg_):
    inputs = tokenizer(msg_, return_tensors="pt", padding=True, truncation=True).to(device)
    input_length = inputs['input_ids'].shape[1]
    with torch.no_grad():  # No need for gradients in inference mode
        output = model.generate(
            inputs['input_ids'],
            max_new_tokens=1,  # You can set this to your desired output length
            return_dict_in_generate=True,
            output_scores=True  # This ensures logits are returned
        )
    results = []
    for i in range(len(msg_)):
        # Extract the generated tokens (including the input tokens)
        generated_tokens = output.sequences[0]
        # Separate the new generated tokens (ignore the initial input tokens)
        new_generated_tokens = generated_tokens[input_length:]
        # Convert new generated token ids back to text
        generated_text = tokenizer.decode(new_generated_tokens, skip_special_tokens=True)
        # Retrieve the logits corresponding to the new generated tokens
        logits_for_new_tokens = torch.stack(output.scores)
        # Only keep logits for the newly generated tokens (after the input length)
        logits_for_new_tokens = logits_for_new_tokens[:len(new_generated_tokens)]
        #print("new_generated_tokens", new_generated_tokens)
        token_id = new_generated_tokens[0].item()
        probabilities = F.softmax(logits_for_new_tokens, dim=-1)
        probability_of_ = probabilities[0, 0, token_id]
        results.append((generated_text, probability_of_.item()))
    return results

saveRepsonses()