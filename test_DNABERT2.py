import os
import sys
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModel

sequence_dir = sys.argv[1]
feature_dir = sys.argv[2]

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).to(device)

max_count = 100
feature_dim = 768
max_length = 1024
name_list = os.listdir(sequence_dir)
for name in name_list:
    sequence_file = sequence_dir + "/" + name
    f = open(sequence_file, "r")
    gene_id = f.readline()[1:].strip()
    sequence = f.readline().strip()
    f.close()
    print(name)
    feature_file = feature_dir + "/" + name.split(".")[0].strip() + ".npy"
    if (os.path.exists(feature_file)):
        continue


    '''

    if(len(sequence)<=cut_off):
        inputs = tokenizer(sequence, return_tensors='pt')["input_ids"].to(device)
        final_hidden_states = model(inputs)[0].to(device)  # [1, sequence_length, 768]

        # embedding with mean pooling

    else:

        inputs = tokenizer(sequence[0:cut_off], return_tensors='pt')["input_ids"].to(device)
        final_hidden_states = model(inputs)[0].to(device)

        sequence = sequence[cut_off:]

        count = 1

        while(len(sequence)>=cut_off):
            inputs = tokenizer(sequence[0:cut_off], return_tensors='pt')["input_ids"].to(device)
            hidden_states = model(inputs)[0].to(device)
            sequence = sequence[cut_off:]
            final_hidden_states = torch.concat([final_hidden_states,hidden_states], dim = 1)
            count = count + 1
            if(count>=max_count):
                break


        if(len(sequence)>0 and count<max_count):
            inputs = tokenizer(sequence, return_tensors='pt')["input_ids"].to(device)
            hidden_states = model(inputs)[0].to(device)
            final_hidden_states = torch.concat([final_hidden_states, hidden_states], dim=1)
    '''

    final_mean_sequence_embeddings = np.zeros(feature_dim)
    count = 0

    while (count == 0 or len(sequence) >= max_length):

        inputs = tokenizer(sequence[0:max_length], return_tensors='pt')["input_ids"].to(device)
        final_hidden_states = model(inputs)[0].to(device)

        final_mean_sequence_embeddings = final_mean_sequence_embeddings + torch.mean(final_hidden_states[0], dim=0).detach().cpu().numpy()
        count = count + 1
        sequence = sequence[max_length:]

        if (count > max_count):
            break

    np.save(feature_file, final_mean_sequence_embeddings/count)




