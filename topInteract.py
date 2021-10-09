#%%
# lzq
# 2021.04.26
# 将蛋白质相互作用概率矩阵中的概率值排序
# 输出概率值最大的蛋白质-蛋白质对到result.csv

# python topInteract.py -i [prob_matrix_file] -o [output path] -n [int]


import argparse

parser = argparse.ArgumentParser(description="Select the most probable interaction")
parser.add_argument("--INPUT",'-i',help="Input file path")
parser.add_argument("--OUTPUT",'-o',help="Output file path")
parser.add_argument("--TOPN",'-n',default=100,type=int,help="The number of interaction selected")
args = parser.parse_args()

BASE_FILE_PATH = args.INPUT
OUTPUT_PATH = args.OUTPUT
TOPN = args.TOPN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def top_prob_interaction(base_file_path, output_path, topn=100):

    prob_score_mat_df = pd.read_csv(base_file_path,index_col=0)
    prob_mat = prob_score_mat_df.values


    N = len(prob_mat)
    for i in range(N):
        prob_mat[i][i:] = np.zeros((N-i))
    prob_values = prob_mat.flatten()
    des_order = np.argsort(prob_values)[::-1]
    interact_A_index = des_order[:topn] // N
    interact_B_index = des_order[:topn] % N
    all_prot = prob_score_mat_df.index
    interact_A = all_prot[interact_A_index]
    interact_B = all_prot[interact_B_index]
    top_probs = prob_values[des_order][:topn]

    df = pd.DataFrame(np.array([interact_A, interact_B, top_probs]).T, columns=['interactor A', 'interactor B', 'prob'])
    df.to_csv(output_path)
    print("Done!")
# 

if __name__ == '__main__':
    top_prob_interaction(BASE_FILE_PATH, OUTPUT_PATH, TOPN)
# %%
