#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MindSpore时间序列预测模型 - 航空乘客数据预测
主程序文件
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# MindSpore相关导入
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter

# 导入自定义模块
from src.data_processor import TimeSeriesDataset
from src.model import LSTMTimeSeriesModel
from src.trainer import TimeSeriesTrainer
from src.evaluator import ModelEvaluator

# 设置MindSpore上下文
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

def create_directories():
    """创建必要的目录"""
    directories = ['data', 'models', 'results', 'src']
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"创建目录: {dir_name}")

def main():
    """主函数"""
    print("=== MindSpore时间序列预测模型 ===")
    
    # 创建目录结构
    create_directories()
    
    # 1. 数据加载和预处理
    print("\n1. 数据加载和预处理...")
    
    # 检查数据文件是否存在
    # data_path = "data/AirPassengers.csv"
    data_path = "data/AirPassengers_USA.csv"
    if not os.path.exists(data_path):
        print(f"警告: 数据文件 {data_path} 不存在，将使用示例数据")
        data_path = None
    
    print("初始化数据集...")
    dataset = TimeSeriesDataset(data_path=data_path, sequence_length=12)
    
    print("准备训练序列...")
    X_train, y_train, X_test, y_test = dataset.prepare_sequences(train_ratio=0.8)
    
    print("数据预处理完成！")
    
    # 显示数据并保存图片
    plt.figure(figsize=(12, 6))
    plt.plot(dataset.data.index, dataset.data['Passengers'])
    plt.title('航空乘客数据 - 时间序列')
    plt.xlabel('日期')
    plt.ylabel('乘客数量')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/original_data.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图片，不显示
    
    # 2. 模型构建
    print("\n2. 模型构建...")
    print("创建LSTM模型...")
    model = LSTMTimeSeriesModel(
        input_size=1,
        hidden_size=50,
        num_layers=2,
        output_size=1,
        dropout=0.2
    )
    
    print(f"模型参数数量: {sum(p.size for p in model.trainable_params())}")
    print("模型构建完成！")
    
    # 3. 模型训练
    print("\n3. 模型训练...")
    print("初始化训练器...")
    trainer = TimeSeriesTrainer(model, learning_rate=0.001)
    
    # 使用部分测试数据作为验证集
    val_split = len(X_test) // 2
    X_val, y_val = X_test[:val_split], y_test[:val_split]
    X_test_final, y_test_final = X_test[val_split:], y_test[val_split:]
    
    print(f"开始训练...")
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test_final)}")
    print(f"批次大小: 16")
    print(f"总训练轮次: 100")
    print("-" * 50)
    
    train_losses, val_losses = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=100, batch_size=16,
        verbose=True  # 添加verbose参数以显示训练进度
    )
    
    # 4. 模型评估
    print("\n4. 模型评估...")
    evaluator = ModelEvaluator(dataset.scaler)
    
    # 训练集预测
    train_pred = trainer.predict(X_train)
    train_metrics = evaluator.calculate_metrics(y_train, train_pred)
    
    # 测试集预测
    test_pred = trainer.predict(X_test_final)
    test_metrics = evaluator.calculate_metrics(y_test_final, test_pred)
    
    print("\n训练集评估结果:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n测试集评估结果:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 5. 结果可视化并保存
    print("\n5. 结果可视化...")
    
    # 保存训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('模型训练历史')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图片，不显示
    
    # 保存预测结果
    y_true_rescaled = dataset.scaler.inverse_transform(y_test_final.reshape(-1, 1)).flatten()
    y_pred_rescaled = dataset.scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_rescaled, label='实际值', linewidth=2)
    plt.plot(y_pred_rescaled, label='预测值', linewidth=2)
    plt.title('测试集预测结果')
    plt.xlabel('时间步')
    plt.ylabel('乘客数量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/predictions.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图片，不显示
    
    # 6. 模型保存
    print("\n6. 模型保存...")
    try:
        ms.save_checkpoint(model, "models/lstm_time_series_model.ckpt")
        print("模型保存成功: models/lstm_time_series_model.ckpt")
    except Exception as e:
        print(f"模型保存失败: {e}")
    
    # 7. 保存评估结果
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model_params': {
            'input_size': 1,
            'hidden_size': 50,
            'num_layers': 2,
            'sequence_length': 12,
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001
        }
    }
    
    import json
    with open('results/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n=== 训练完成 ===")
    print("结果文件已保存到 results/ 目录")
    
    # 8. 未来预测示例
    print("\n=== 未来预测示例 ===")
    last_sequence = dataset.scaled_data[-12:].reshape(1, 12, 1)
    future_pred = trainer.model(Tensor(last_sequence, ms.float32))
    future_value = dataset.scaler.inverse_transform(future_pred.asnumpy().reshape(-1, 1))
    
    print(f"下一期预测值: {future_value[0][0]:.2f}")
    
    return model, dataset, trainer, evaluator


if __name__ == "__main__":
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("警告: 建议使用Python 3.8或更高版本")
    
    # 运行主程序
    try:
        model, dataset, trainer, evaluator = main()
        
        print("\n程序执行成功！")
        print("查看 results/ 目录获取详细结果")
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()