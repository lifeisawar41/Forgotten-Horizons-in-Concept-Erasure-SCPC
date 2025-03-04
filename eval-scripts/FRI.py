import pandas as pd
import numpy as np


def calculate_fri(sheet_data, method):
    """
    计算指定方法的 FRI 指标。
    :param sheet_data: 一个 sheet 的数据（DataFrame）
    :param method: 方法名称（'ga', 'esd', 'salun', 'ours'）
    :return: FRI 值
    """
    # 检查列名
    # print("列名:", sheet_data.columns)

    sheet_data.columns = sheet_data.columns.str.strip()

    target_row = sheet_data.iloc[-1]
    target_clipscore_original = target_row['ori sd1.4']
    target_clipscore_forgotten = float(target_row[method]) 

    target_diff = target_clipscore_forgotten - target_clipscore_original
    target_std = np.abs(target_diff)

    # 提取邻近类别数据
    neighbor_rows = sheet_data.iloc[:-1]
    text_similarities = neighbor_rows['text similarity']  
    clipscore_original = neighbor_rows['ori sd1.4']
    clipscore_forgotten = neighbor_rows[method].astype(float)  

    weights = text_similarities / text_similarities.sum()

    neighbor_diffs = clipscore_forgotten - clipscore_original
    neighbor_stds = np.abs(neighbor_diffs) 
    weighted_neighbor_std = np.sum(weights * neighbor_stds)

    fri = target_std - weighted_neighbor_std
    return fri

file_path = "/sdb4/case/ly/US-SD/eval-scripts/scpc-clipscore_new3.2.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None)

# 计算每种方法的 FRI
results = {}
for sheet_name, sheet_data in sheets.items():
    fri_ga = calculate_fri(sheet_data, 'ga')
    fri_esd = calculate_fri(sheet_data, 'esd')
    fri_salun = calculate_fri(sheet_data, 'salun')
    fri_ours = calculate_fri(sheet_data, 'ours')
    results[sheet_name] = {
        'ga': fri_ga,
        'esd': fri_esd,
        'salun': fri_salun,
        'ours': fri_ours
    }

for sheet_name, fri_values in results.items():
    print(f"Sheet: {sheet_name}")
    print(f"FRI (ga): {fri_values['ga']:.4f}")
    print(f"FRI (esd): {fri_values['esd']:.4f}")
    print(f"FRI (salun): {fri_values['salun']:.4f}")
    print(f"FRI (ours): {fri_values['ours']:.4f}")
    print("-" * 30)