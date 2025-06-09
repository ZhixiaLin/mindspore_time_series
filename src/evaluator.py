#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, scaler=None):
        """
        初始化评估器
        
        Args:
            scaler: 数据标准化器，用于反标准化
        """
        self.scaler = scaler
        self.metrics_history = []
    
    def calculate_metrics(self, y_true, y_pred, scaled=True):
        """
        计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            scaled: 数据是否经过标准化
            
        Returns:
            dict: 包含各种评估指标的字典
        """
        # 如果数据经过标准化，先反标准化
        if scaled and self.scaler is not None:
            y_true_orig = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_orig = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_true_orig = y_true.flatten()
            y_pred_orig = y_pred.flatten()
        
        # 计算各种指标
        mae = mean_absolute_error(y_true_orig, y_pred_orig)
        mse = mean_squared_error(y_true_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_orig, y_pred_orig)
        
        # 计算MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true_orig - y_pred_orig) / np.maximum(np.abs(y_true_orig), 1e-8))) * 100
        
        # 计算方向准确率
        if len(y_true_orig) > 1:
            true_direction = np.diff(y_true_orig) > 0
            pred_direction = np.diff(y_pred_orig) > 0
            direction_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            direction_accuracy = 0
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy
        }
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, title="预测结果对比", save_path=None):
        """
        绘制预测结果对比图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            save_path: 保存路径
        """
        # 反标准化
        if self.scaler is not None:
            y_true_plot = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_plot = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_true_plot = y_true.flatten()
            y_pred_plot = y_pred.flatten()
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true_plot, label='实际值', linewidth=2, alpha=0.8)
        plt.plot(y_pred_plot, label='预测值', linewidth=2, alpha=0.8)
        plt.title(title)
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_scatter(self, y_true, y_pred, title="散点图对比", save_path=None):
        """
        绘制散点图对比真实值和预测值
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            save_path: 保存路径
        """
        # 反标准化
        if self.scaler is not None:
            y_true_plot = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_plot = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_true_plot = y_true.flatten()
            y_pred_plot = y_pred.flatten()
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true_plot, y_pred_plot, alpha=0.6)
        
        # 添加完美预测线
        min_val = min(np.min(y_true_plot), np.min(y_pred_plot))
        max_val = max(np.max(y_true_plot), np.max(y_pred_plot))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')
        
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, title="残差分析", save_path=None):
        """
        绘制残差分析图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            save_path: 保存路径
        """
        # 反标准化
        if self.scaler is not None:
            y_true_plot = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_plot = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_true_plot = y_true.flatten()
            y_pred_plot = y_pred.flatten()
        
        residuals = y_true_plot - y_pred_plot
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 残差时间序列图
        axes[0, 0].plot(residuals)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('残差时间序列')
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('残差')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 残差分布直方图
        axes[0, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('残差分布')
        axes[0, 1].set_xlabel('残差')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 残差 vs 预测值散点图
        axes[1, 0].scatter(y_pred_plot, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('残差 vs 预测值')
        axes[1, 0].set_xlabel('预测值')
        axes[1, 0].set_ylabel('残差')
        axes[1, 0].grid(True, alpha=0.3)
        
        # QQ图（正态性检验）
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q图（正态性检验）')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_distribution(self, y_true, y_pred, save_path=None):
        """
        绘制误差分布图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            save_path: 保存路径
        """
        # 反标准化
        if self.scaler is not None:
            y_true_plot = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_plot = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_true_plot = y_true.flatten()
            y_pred_plot = y_pred.flatten()
        
        # 计算各种误差
        absolute_error = np.abs(y_true_plot - y_pred_plot)
        percentage_error = np.abs((y_true_plot - y_pred_plot) / np.maximum(np.abs(y_true_plot), 1e-8)) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绝对误差分布
        axes[0].hist(absolute_error, bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_title('绝对误差分布')
        axes[0].set_xlabel('绝对误差')
        axes[0].set_ylabel('频次')
        axes[0].grid(True, alpha=0.3)
        
        # 百分比误差分布
        axes[1].hist(percentage_error, bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_title('百分比误差分布')
        axes[1].set_xlabel('百分比误差 (%)')
        axes[1].set_ylabel('频次')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def comprehensive_evaluation(self, y_true, y_pred, dataset_name="", save_dir=None):
        """
        综合评估
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            dataset_name: 数据集名称
            save_dir: 保存目录
            
        Returns:
            dict: 评估指标
        """
        print(f"\n=== {dataset_name} 综合评估结果 ===")
        
        # 计算指标
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # 打印指标
        for metric, value in metrics.items():
            if metric == 'MAPE':
                print(f"{metric}: {value:.2f}%")
            elif metric == 'Direction_Accuracy':
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value:.4f}")
        
        # 保存图表
        if save_dir:
            self.plot_predictions(y_true, y_pred, 
                                f"{dataset_name} 预测结果对比",
                                f"{save_dir}/{dataset_name.lower()}_predictions.png")
            
            self.plot_scatter(y_true, y_pred, 
                            f"{dataset_name} 散点图对比",
                            f"{save_dir}/{dataset_name.lower()}_scatter.png")
            
            self.plot_residuals(y_true, y_pred, 
                              f"{dataset_name} 残差分析",
                              f"{save_dir}/{dataset_name.lower()}_residuals.png")
            
            self.plot_error_distribution(y_true, y_pred,
                                       f"{save_dir}/{dataset_name.lower()}_error_dist.png")
        
        # 保存指标历史
        metrics['dataset'] = dataset_name
        self.metrics_history.append(metrics)
        
        return metrics
    
    def compare_models(self, results_dict, save_path=None):
        """
        比较多个模型的性能
        
        Args:
            results_dict: 模型结果字典 {model_name: (y_true, y_pred)}
            save_path: 保存路径
        """
        metrics_df = []
        
        for model_name, (y_true, y_pred) in results_dict.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics['Model'] = model_name
            metrics_df.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_df)
        
        # 显示比较表
        print("\n=== 模型性能比较 ===")
        print(metrics_df.set_index('Model'))
        
        # 绘制比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics_to_plot = ['MAE', 'RMSE', 'MAPE', 'R2']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//2, i%2]
            metrics_df.plot(x='Model', y=metric, kind='bar', ax=ax)
            ax.set_title(f'{metric} 比较')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics_df
    
    def generate_report(self, metrics, model_info=None, save_path=None):
        """
        生成评估报告
        
        Args:
            metrics: 评估指标字典
            model_info: 模型信息
            save_path: 保存路径
        """
        report = []
        report.append("# 时间序列预测模型评估报告\n")
        
        if model_info:
            report.append("## 模型信息")
            for key, value in model_info.items():
                report.append(f"- {key}: {value}")
            report.append("")
        
        report.append("## 评估指标")
        for metric, value in metrics.items():
            if metric == 'MAPE':
                report.append(f"- {metric}: {value:.2f}%")
            elif metric == 'Direction_Accuracy':
                report.append(f"- {metric}: {value:.2f}%")
            else:
                report.append(f"- {metric}: {value:.4f}")
        
        report.append("\n## 指标说明")
        report.append("- MAE (Mean Absolute Error): 平均绝对误差，越小越好")
        report.append("- MSE (Mean Squared Error): 均方误差，越小越好")
        report.append("- RMSE (Root Mean Squared Error): 均方根误差，越小越好")
        report.append("- R² (R-squared): 决定系数，越接近1越好")
        report.append("- MAPE (Mean Absolute Percentage Error): 平均绝对百分比误差，越小越好")
        report.append("- Direction Accuracy: 方向准确率，预测趋势的准确性，越高越好")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"评估报告已保存到: {save_path}")
        
        return report_text