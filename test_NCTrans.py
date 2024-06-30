import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np

sequence_dir = sys.argv[1]
feature_dir = sys.argv[2]
feature_dim = 2560

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species").to(device)

# Choose the length to which the input sequences are padded. By default, the
# model max length is chosen, but feel free to decrease it as the time taken to
# obtain the embeddings increases significantly with it.
max_length = tokenizer.model_max_length//2

# Create a dummy dna sequence and tokenize it

name_list = os.listdir(sequence_dir)
for name in name_list:

    sequence_file = sequence_dir + "/" + name
    f = open(sequence_file, "r")
    gene_id = f.readline()[1:].strip()
    sequence = f.readline().strip()
    f.close()

    feature_file = feature_dir + "/" + name.split(".")[0].strip() + ".npy"
    if (os.path.exists(feature_file)):
        continue

    final_mean_sequence_embeddings = np.zeros(feature_dim)
    count = 0

    while(count ==0 or len(sequence)>=max_length):

        tokens_ids = tokenizer.batch_encode_plus([sequence[0:max_length]], return_tensors="pt", padding="max_length",max_length=max_length)["input_ids"].to(device)

        # Compute the embeddings
        attention_mask = tokens_ids != tokenizer.pad_token_id
        torch_outs = model(
            tokens_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )

        embeddings = torch_outs['hidden_states'][-1]

        attention_mask = torch.unsqueeze(attention_mask, dim=-1)

        mean_sequence_embeddings = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(attention_mask, axis=1)
        final_mean_sequence_embeddings = final_mean_sequence_embeddings + mean_sequence_embeddings[0].detach().cpu().numpy()
        count = count + 1
        sequence = sequence[max_length:]

        if(count>200):
            break

    np.save(feature_file, final_mean_sequence_embeddings/count)
    torch.cuda.empty_cache()



