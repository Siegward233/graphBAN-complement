import torch
import numpy as np
import pandas as pd
import argparse
import os

import copy

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaModel
import esm
from models import GraphBAN
from utils import graph_collate_func, mkdir
from dataloader import DTIDataset
from configs import get_cfg_defaults
from trainer_pred import Trainer

# Argument parser
parser = argparse.ArgumentParser(description="Batch prediction for multiple test files.")
parser.add_argument("--data_dir", type=str, required=True, help="Directory containing split_zinc_*.csv files.")

parser.add_argument("--model_1_path", type=str, help='Path to the .pth files')
parser.add_argument("--model_2_path", type=str, help='Path to the .pth files')
parser.add_argument("--model_3_path", type=str, help='Path to the .pth files')

parser.add_argument("--trained_model_name", type=str, help='model name')
parser.add_argument("--top_number", type=int, help='top smiles number that you want')
parser.add_argument("--folder_path", type=str, help='Path to save result')
parser.add_argument("--save_dir", type=str, required=True, help="Path to save the top100 result CSV.")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ESM model
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
esm_model = esm_model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

def Get_Protein_Feature(p_list):
    data_tmp = [(f"protein{i}", p[:1022]) for i, p in enumerate(p_list)]
    dictionary = {}
    for i in range((len(data_tmp) + 4) // 5):
        data_part = data_tmp[i * 5:(i + 1) * 5]
        _, _, batch_tokens = batch_converter(data_part)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        for j, (_, seq) in enumerate(data_part):
            emb_rep = token_representations[j, 1:len(seq) + 1].mean(0).cpu().numpy()
            dictionary[seq] = emb_rep
    return pd.DataFrame(dictionary.items(), columns=['Protein', 'esm'])

# Load ChemBERTa
model_name = "DeepChem/ChemBERTa-77M-MTR"
model_chem = RobertaModel.from_pretrained(model_name, num_labels=2, add_pooling_layer=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_embeddings(df):
    emb_list = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):
        encodings = tokenizer(row['SMILES'], return_tensors='pt', padding="max_length", max_length=290, truncation=True).to(device)
        with torch.no_grad():
            output = model_chem(**encodings)
            smiles_embeddings = output.last_hidden_state[0, 0].cpu().numpy()
            emb_list.append(smiles_embeddings)
    return emb_list

# Load config and model
cfg = get_cfg_defaults()
cfg.merge_from_file("case_study/GraphBAN_DA.yaml")
cfg.freeze()
mkdir(args.folder_path)

modelG = GraphBAN(**cfg).to(device)
opt = torch.optim.Adam(modelG.parameters(), lr=cfg.SOLVER.LR)
model_paths = [args.model_1_path, args.model_2_path, args.model_3_path]
#models = []

# 加载三个模型
#for path in model_paths:
#    model = copy.deepcopy(modelG)  # 确保每个模型是独立的副本
#    model.load_state_dict(torch.load(path))
#    models.append(model)

final_results = []

for i in range(1, 26):
    test_path = os.path.join(args.data_dir, f"split_zinc_{i}.csv")
    print(f"Processing {test_path}...")
    df_test = pd.read_csv(test_path)
    df_test['Protein'] = df_test['Protein'].apply(lambda x: x[:1022] if len(x) > 1022 else x)

    pro_list_test = df_test['Protein'].unique()
    df_test = pd.merge(df_test, Get_Protein_Feature(list(pro_list_test)), on='Protein', how='left')

    df_test_unique = df_test.drop_duplicates(subset='SMILES')
    emb_list_test = get_embeddings(df_test_unique)
    df_test_unique['fcfp'] = emb_list_test
    df_test = pd.merge(df_test, df_test_unique[['SMILES', 'fcfp']], on='SMILES', how='left')

    test_dataset = DTIDataset(df_test.index.values, df_test)
    test_generator = DataLoader(test_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                 num_workers=cfg.SOLVER.NUM_WORKERS, drop_last=False, collate_fn=graph_collate_func)

    # 对每个模型进行预测
    
    all_preds = pd.DataFrame()
    i = 0
    for model in model_paths:
        modelG.load_state_dict(torch.load(model))
        trainer = Trainer(modelG, opt, device, test_generator, **cfg)
        pred = trainer.train()
        print(f"no.{i}model predicting")
        all_preds[f'pred{i}'] = pred
        i = i+1

    # 对预测结果取平均
    #avg_pred = sum(all_preds) / len(all_preds)
    all_preds['row_average'] = all_preds.mean(axis=1)
    
    print(all_preds.head())

    df_test['predicted_value'] = all_preds['row_average']

    df_result = df_test[['SMILES', 'Protein', 'predicted_value']]
    final_results.append(df_result)

# 合并所有结果并筛选
df_all = pd.concat(final_results, ignore_index=True)
df_top100 = df_all[df_all['predicted_value'] > 0.5]
df_top100.to_csv(args.save_dir, index=False)

print(f"{args.trained_model_name} Top {args.top_number} results saved to {args.save_dir}")
