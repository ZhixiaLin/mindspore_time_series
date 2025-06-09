#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练器模块
"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import time


class TimeSeriesTrainer:
    """时间序列模型训练器"""
    
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-4):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 定义损失函数和优化器
        self.loss_fn = nn.MSELoss()
        self.optimizer = nn.Adam(
            model.trainable_params(), 
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # 定义前向函数
        def forward_fn(data, label):
            logits = self.model(data)
            loss = self.loss_fn(logits, label)
            return loss, logits
        
        # 定义梯度函数
        self.grad_fn = ops.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, data, label):
        """
        单步训练
        
        Args:
            data: 输入数据
            label: 标签数据
            
        Returns:
            loss: 损失值
        """
        (loss, _), grads = self.grad_fn(data, label)
        self.optimizer(grads)
        return loss
    
    def validate_step(self, data, label):
        """
        验证步骤
        
        Args:
            data: 输入数据
            label: 标签数据
            
        Returns:
            loss: 验证损失
        """
        self.model.set_train(False)
        logits = self.model(data)
        loss = self.loss_fn(logits, label)
        return loss
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=True):
        """
        训练模型
        
        Args:
            X_train: 训练输入数据
            y_train: 训练标签数据
            X_val: 验证输入数据
            y_val: 验证标签数据
            epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否打印训练信息
            
        Returns:
            train_losses: 训练损失历史
            val_losses: 验证损失历史
        """
        print("开始训练模型...")
        
        # 将数据转换为Tensor
        X_train = Tensor(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), ms.float32)
        y_train = Tensor(y_train.reshape(-1, 1), ms.float32)
        
        if X_val is not None and y_val is not None:
            X_val = Tensor(X_val.reshape(X_val.shape[0], X_val.shape[1], 1), ms.float32)
            y_val = Tensor(y_val.reshape(-1, 1), ms.float32)
        
        self.train_losses = []
        self.val_losses = []
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练模式
            self.model.set_train()
            
            # 批量训练
            total_loss = 0
            num_batches = 0
            
            # 显示进度条
            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print("训练中...", end="")
            
            for i in range(0, len(X_train), batch_size):
                end_idx = min(i + batch_size, len(X_train))
                batch_x = X_train[i:end_idx]
                batch_y = y_train[i:end_idx]
                
                loss = self.train_step(batch_x, batch_y)
                total_loss += loss.asnumpy()
                num_batches += 1
                
                # 显示进度点
                if verbose and i % (batch_size * 5) == 0:
                    print(".", end="", flush=True)
            
            avg_train_loss = total_loss / max(num_batches, 1)
            self.train_losses.append(avg_train_loss)
            
            # 验证
            if X_val is not None and y_val is not None:
                self.model.set_train(False)
                val_loss = self.validate_step(X_val, y_val)
                val_loss_value = val_loss.asnumpy()
                self.val_losses.append(val_loss_value)
                
                # 早停检查
                if val_loss_value < best_val_loss:
                    best_val_loss = val_loss_value
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose:
                    print(f"\nEpoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {val_loss_value:.6f}")
                
                # 早停
                if patience_counter >= patience:
                    print(f"\n早停在第 {epoch+1} 轮")
                    break
            else:
                if verbose:
                    print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n训练完成！耗时: {training_time:.2f}秒")
        print(f"最终训练损失: {self.train_losses[-1]:.6f}")
        if self.val_losses:
            print(f"最终验证损失: {self.val_losses[-1]:.6f}")
        
        return self.train_losses, self.val_losses
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 输入数据
            
        Returns:
            predictions: 预测结果
        """
        self.model.set_train(False)
        
        if not isinstance(X, Tensor):
            X = Tensor(X.reshape(X.shape[0], X.shape[1], 1), ms.float32)
        
        predictions = self.model(X)
        return predictions.asnumpy()
    
    def save_model(self, save_path):
        """保存模型"""
        ms.save_checkpoint(self.model, save_path)
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path):
        """加载模型"""
        param_dict = ms.load_checkpoint(load_path)
        ms.load_param_into_net(self.model, param_dict)
        print(f"模型已从 {load_path} 加载")


class EarlyStopping:
    """早停回调"""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        """
        初始化早停
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        """
        检查是否需要早停
        
        Args:
            val_loss: 验证损失
            model: 模型
            
        Returns:
            bool: 是否需要停止训练
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = model.parameters_dict()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.parameters_dict()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                # 恢复最佳权重（在MindSpore中需要手动实现）
                pass
            return True
        
        return False


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器
            mode: 模式 ('min' 或 'max')
            factor: 衰减因子
            patience: 容忍轮数
            min_lr: 最小学习率
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = None
        self.counter = 0
        self.current_lr = optimizer.learning_rate.asnumpy()
    
    def step(self, val_loss):
        """
        更新学习率
        
        Args:
            val_loss: 验证损失
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif (self.mode == 'min' and val_loss >= self.best_loss) or \
             (self.mode == 'max' and val_loss <= self.best_loss):
            self.counter += 1
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        if self.counter >= self.patience:
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            if new_lr != self.current_lr:
                self.current_lr = new_lr
                # 在MindSpore中更新学习率需要重新创建优化器或使用动态学习率
                print(f"学习率调整为: {new_lr}")
            self.counter = 0