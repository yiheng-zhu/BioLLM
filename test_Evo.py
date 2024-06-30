import os
import sys
import numpy as np
from evo import Evo

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch

sequence_dir = sys.argv[1]
feature_dir = sys.argv[2]

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

evo_model = Evo('evo-1-8k-base')
model, tokenizer = evo_model.model, evo_model.tokenizer
model.to(device)
model.eval()


max_count = 500
feature_dim = 512
max_length = 256
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

    final_mean_sequence_embeddings = np.zeros(feature_dim)
    count = 0

    try:
        while (count == 0 or len(sequence) >= max_length):

            input_ids = torch.tensor(tokenizer.tokenize(sequence[0:max_length]), dtype=torch.int, ).to(
                device).unsqueeze(0)
            final_hidden_states, _ = model(input_ids)  # (batch, length, vocab)

            final_mean_sequence_embeddings = final_mean_sequence_embeddings + torch.mean(final_hidden_states[0],
                                                                                         dim=0).detach().to(
                torch.float).cpu().numpy()
            count = count + 1
            sequence = sequence[max_length:]

            if (count > max_count):
                break

        np.save(feature_file, final_mean_sequence_embeddings / count)
    except Exception as e:
        print("Out of memory")

