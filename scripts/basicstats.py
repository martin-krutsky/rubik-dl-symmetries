import ast
import pandas as pd
import numpy as np

dfkoc = pd.read_csv(f'data/processed/kociemba_dataset.csv', index_col=0, converters={'colors':ast.literal_eval, 'generator':ast.literal_eval})
for i in range(14,20):
    filtered_dfkoc = dfkoc[dfkoc['distance'] == i]
    nr_classes = len(filtered_dfkoc['class_id'].unique())
    avg_compr = len(filtered_dfkoc) / nr_classes
    print(f'DISTANCE {i}')
    print(f'  nr of states: {len(filtered_dfkoc)}')
    print(f'  nr of classes: {nr_classes}')
    print(f'  average compression ratio: {avg_compr}')
    
df6m = pd.read_csv(f'data/processed/6_moves_dataset_single.csv', index_col=0, converters={'colors':ast.literal_eval, 'generator':ast.literal_eval})
for i in range(1,7):
    filtered_df6m = df6m[df6m['distance'] == i]
    nr_classes = len(filtered_df6m['class_id'].unique())
    avg_compr = len(filtered_df6m) / nr_classes
    print(f'DISTANCE {i}')
    print(f'  nr of states: {len(filtered_df6m)}')
    print(f'  nr of classes: {nr_classes}')
    print(f'  average compression ratio: {avg_compr}')
