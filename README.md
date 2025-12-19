# 虚假新闻检测与分级治理系统
该项目实现了一个集成化的虚假新闻检测与分级治理系统，结合文本特征分析、传播特征建模和用户特征挖掘，实现对新闻内容的风险评估与分级处置。

## 功能说明
- **数据预处理**：加载 WELFake 数据集并进行清洗、领域标注和特征生成
- **算法助推机理量化**：分析不同推荐算法对虚假新闻传播的助推效应
- **多特征融合预警模型**：构建集成模型实现虚假新闻检测
- **分级治理与协同联动**：根据风险等级执行不同的治理策略并模拟效果
  
## 环境依赖
```bash
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
torch>=1.8.0
transformers>=4.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.60.0
joblib>=1.0.0
```
## 使用方法
直接运行主程序：
```bash
python python main.py
```
程序将自动执行以下流程：
- 数据预处理并生成data/processed_welfake.csv
- 分析算法助推效应并生成可视化结果
- 训练集成预警模型并保存到models/目录
- 对测试新闻进行风险评估和治理策略推荐
- 模拟治理效果并输出统计结果

## 输出文件
程序运行后将生成以下文件：
- **数据文件**
-data/processed_welfake.csv: 预处理后的完整数据集
- **结果文件**
- results/boost_coefficients.csv: 算法 - 领域组合的助推系数
- results/boost_coeff_heatmap.png: 助推系数热力图
- results/performance_comparison.png: 模型性能对比图
- 
- **模型文件**
- models/scaler.pkl: 特征标准化器
- models/spread_model.pth: 传播特征模型
- models/text_rf.pkl: 文本特征随机森林模型
- models/user_rf.pkl: 用户特征随机森林模型

## 注意事项
- 首次运行会自动下载 BERT 预训练模型，可能需要较长时间
- 在GPU 的环境下运行
- 数据集较大时，预处理和模型训练过程可能需要较长时间
