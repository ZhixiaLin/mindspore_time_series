#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM时间序列预测模型定义
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter


class LSTMTimeSeriesModel(nn.Cell):
    """LSTM时间序列预测模型"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: Dropout比例
        """
        super(LSTMTimeSeriesModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Dense(hidden_size, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def construct(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_size)
            
        Returns:
            output: 输出张量，形状为 (batch_size, output_size)
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # 全连接层
        output = self.fc(lstm_out)
        
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.size for p in self.trainable_params())
        
        info = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'total_parameters': total_params
        }
        
        return info


class GRUTimeSeriesModel(nn.Cell):
    """GRU时间序列预测模型 (可选的替代模型)"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        """
        初始化GRU模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: GRU隐藏层大小
            num_layers: GRU层数
            output_size: 输出维度
            dropout: Dropout比例
        """
        super(GRUTimeSeriesModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Dense(hidden_size, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def construct(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_size)
            
        Returns:
            output: 输出张量，形状为 (batch_size, output_size)
        """
        # GRU前向传播
        gru_out, _ = self.gru(x)
        
        # 取最后一个时间步的输出
        gru_out = gru_out[:, -1, :]
        
        # Dropout
        gru_out = self.dropout(gru_out)
        
        # 全连接层
        output = self.fc(gru_out)
        
        return output


class SimpleRNNModel(nn.Cell):
    """简单RNN模型 (用于对比实验)"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        """
        初始化简单RNN模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: RNN隐藏层大小
            num_layers: RNN层数
            output_size: 输出维度
            dropout: Dropout比例
        """
        super(SimpleRNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN层
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Dense(hidden_size, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def construct(self, x):
        """前向传播"""
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]
        rnn_out = self.dropout(rnn_out)
        output = self.fc(rnn_out)
        return output


def create_model(model_type='lstm', **kwargs):
    """
    模型工厂函数
    
    Args:
        model_type: 模型类型 ('lstm', 'gru', 'rnn')
        **kwargs: 模型参数
        
    Returns:
        model: 创建的模型实例
    """
    if model_type.lower() == 'lstm':
        return LSTMTimeSeriesModel(**kwargs)
    elif model_type.lower() == 'gru':
        return GRUTimeSeriesModel(**kwargs)
    elif model_type.lower() == 'rnn':
        return SimpleRNNModel(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def print_model_summary(model):
    """打印模型摘要"""
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print("\n=== 模型信息 ===")
        for key, value in info.items():
            print(f"{key}: {value}")
    else:
        total_params = sum(p.size for p in model.trainable_params())
        print(f"\n总参数量: {total_params}")