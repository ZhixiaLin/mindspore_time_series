#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块
处理时间序列数据的加载、预处理和序列生成
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesDataset:
    """时间序列数据处理类"""
    
    def __init__(self, data_path=None, sequence_length=12):
        """
        初始化数据处理器
        
        Args:
            data_path: 数据文件路径
            sequence_length: 输入序列长度
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.data = None
        self.scaled_data = None
        
        if data_path:
            self.load_data(data_path)
        else:
            self.create_sample_data()
    
    def load_data(self, data_path):
        """加载航空乘客数据"""
        try:
            # 读取CSV文件
            self.data = pd.read_csv(data_path)
            
            # 数据预处理 - 处理不同的列名格式
            if 'Month' in self.data.columns and '#Passengers' in self.data.columns:
                self.data['Month'] = pd.to_datetime(self.data['Month'])
                self.data.set_index('Month', inplace=True)
                self.data.columns = ['Passengers']
            elif 'Month' in self.data.columns and 'Passengers' in self.data.columns:
                self.data['Month'] = pd.to_datetime(self.data['Month'])
                self.data.set_index('Month', inplace=True)
            elif len(self.data.columns) == 2:
                # 假设第一列是日期，第二列是乘客数
                self.data.columns = ['Month', 'Passengers']
                self.data['Month'] = pd.to_datetime(self.data['Month'])
                self.data.set_index('Month', inplace=True)
            
            # 确保数据是数值型
            self.data['Passengers'] = pd.to_numeric(self.data['Passengers'], errors='coerce')
            
            # 处理缺失值
            self.data.dropna(inplace=True)
            
            print(f"数据加载成功，共{len(self.data)}个数据点")
            print(f"数据范围: {self.data.index[0]} 到 {self.data.index[-1]}")
            print(f"数据统计:\n{self.data.describe()}")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            print("使用示例数据...")
            self.create_sample_data()
    
    def create_sample_data(self):
        """创建示例航空乘客数据"""
        print("创建示例数据...")
        dates = pd.date_range('1949-01', '1960-12', freq='M')
        
        # 模拟航空乘客数据的季节性和趋势
        trend = np.linspace(100, 600, len(dates))
        seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 20, len(dates))
        passengers = trend + seasonal + noise
        passengers = np.maximum(passengers, 50)  # 确保非负
        
        self.data = pd.DataFrame({'Passengers': passengers}, index=dates)
        print(f"示例数据创建完成，共{len(self.data)}个数据点")
        print(f"数据统计:\n{self.data.describe()}")
    
    def plot_data(self):
        """数据可视化"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Passengers'])
        plt.title('航空乘客数据 - 时间序列')
        plt.xlabel('日期')
        plt.ylabel('乘客数量')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def analyze_seasonality(self):
        """分析季节性"""
        # 按月份分组
        monthly_avg = self.data.groupby(self.data.index.month).mean()
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, 13), monthly_avg['Passengers'])
        plt.title('各月份平均乘客数量')
        plt.xlabel('月份')
        plt.ylabel('平均乘客数量')
        plt.xticks(range(1, 13))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return monthly_avg
    
    def prepare_sequences(self, train_ratio=0.8):
        """
        准备训练序列
        
        Args:
            train_ratio: 训练集比例
            
        Returns:
            X_train, y_train, X_test, y_test: 训练和测试数据
        """
        # 数据标准化
        self.scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1, 1))
        
        # 创建序列
        X, y = [], []
        for i in range(self.sequence_length, len(self.scaled_data)):
            X.append(self.scaled_data[i-self.sequence_length:i, 0])
            y.append(self.scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # 划分训练集和测试集
        split_idx = int(len(X) * train_ratio)
        
        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_test = X[split_idx:]
        self.y_test = y[split_idx:]
        
        print(f"序列长度: {self.sequence_length}")
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def get_last_sequence(self):
        """获取最后一个序列用于预测"""
        if self.scaled_data is None:
            raise ValueError("请先调用 prepare_sequences 方法")
        
        return self.scaled_data[-self.sequence_length:]
    
    def inverse_transform(self, scaled_values):
        """反标准化"""
        return self.scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()