"""
预处理美国航空乘客数据（Kaggle下载）
原始数据来源：https://www.kaggle.com/datasets/ramjasmaurya/us-domestic-airline-flights

- 默认统计每月总乘客数，输出为 Month,#Passengers 格式
- 支持按 Terminal、Boarding Area 等字段分组统计，便于后续扩展
"""
import pandas as pd
import os

def preprocess_air_traffic(input_file, output_dir, group_by=None):
    """
    预处理航空乘客数据
    Args:
        input_file: 原始CSV文件路径
        output_dir: 输出目录
        group_by: list, 按哪些字段分组（如 ['Terminal']），None为总乘客数
    """
    df = pd.read_csv(input_file)
    # 检查字段
    required_cols = {'Year', 'Month', 'Passenger Count'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"缺少必要字段: {required_cols - set(df.columns)}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if group_by is None:
        # 统计每月总乘客数
        monthly = df.groupby(['Year', 'Month'])['Passenger Count'].sum().reset_index()
        monthly['Month'] = monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2)
        monthly = monthly[['Month', 'Passenger Count']]
        monthly.columns = ['Month', '#Passengers']
        out_file = os.path.join(output_dir, 'AirPassengers_USA.csv')
        monthly.to_csv(out_file, index=False)
        print(f'已保存: {out_file}')
    else:
        # 按指定字段分组统计
        grouped = df.groupby(group_by + ['Year', 'Month'])['Passenger Count'].sum().reset_index()
        grouped['Month'] = grouped['Year'].astype(str) + '-' + grouped['Month'].astype(str).str.zfill(2)
        for keys, subdf in grouped.groupby(group_by):
            if not isinstance(keys, tuple):
                keys = (keys,)
            group_name = '_'.join([str(k) for k in keys])
            out_file = os.path.join(output_dir, f'AirPassengers_USA_{group_name}.csv')
            out = subdf[['Month', 'Passenger Count']].copy()
            out.columns = ['Month', '#Passengers']
            out.to_csv(out_file, index=False)
            print(f'已保存: {out_file}')

if __name__ == '__main__':
    # 默认统计总乘客数
    preprocess_air_traffic(
        input_file='data/Air_Traffic_Passenger_Statistics.csv',
        output_dir='data',
        group_by=None
    )
    # 预留分组分析接口（如需分组，取消注释并指定字段）
    # preprocess_air_traffic(
    #     input_file='../data/Air_Traffic_Passenger_Statistics.csv',
    #     output_dir='../data',
    #     group_by=['Terminal']
    # ) 