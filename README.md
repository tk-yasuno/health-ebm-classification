# 🏗️ Health EBM Classification

**Bridge Health Level Classification using Explainable Boosting Machine**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready%20v0.3-success.svg)]()

## 🎯 Overview

An AI-powered bridge health classification system that automatically categorizes bridge inspection reports into health levels using machine learning. The system leverages **Explainable Boosting Machine (EBM)** to achieve high accuracy while maintaining interpretability.

### 🏆 Key Achievements (v0.3)
- **🚀 EBM 25倍高速化**: 25分 → 63.8秒（16並列処理）
- **91.88% Test Accuracy** - 実用レベル達成！
- **86.20% F1-macro score** (+19.8pt improvement from v0.2)
- **フルデータ学習**: 8,615件処理（31倍データ活用）
- **Repair-requirement F1: 80%** - 実用性確保

### 📊 Version History
| Version | Data Size | Test Accuracy | F1-Macro | Key Innovation |
|---------|-----------|---------------|----------|----------------|
| **v0.3** | **8,615件** | **91.88%** | **86.20%** | **16並列高速化 + フルデータ** |
| v0.2 | 276件 | 84.34% | 66.40% | 集約データでのMVP |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/health-ebm-classification.git
cd health-ebm-classification

# Set up virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate    # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
cd src
python main_pipeline.py
```

## 📊 Performance Results (v0.3)

| Model | Training Time | Val F1-Macro | Test Accuracy | 特徴 |
|-------|---------------|-------------|---------------|------|
| **🥇 EBM** | **63.80秒** | **85.34%** | **91.88%** | **最高精度+高速化** |
| 🥈 XGBoost Enhanced | 5.42秒 | 82.91% | 89.12% | 高速高精度 |
| 🥉 CatBoost | 28.12秒 | 79.44% | 87.56% | バランス型 |
| LightGBM | 2.23秒 | 76.38% | 85.23% | 超高速 |
| Random Forest | 0.42秒 | 71.64% | 82.34% | 最高速 |

### 🔥 v0.3 革命的改善
- **EBM高速化**: 25分 → 63.8秒（**25倍高速化**）
- **精度向上**: Test Accuracy 84.34% → **91.88%** (+7.54pt)
- **F1向上**: 66.40% → **86.20%** (+19.80pt)
- **実用性**: Repair-requirement F1 25% → **80%** (+55pt)

## 🏗️ Architecture

### Data Pipeline (v0.3)
```
Raw CSV Data → Preprocessing → Feature Engineering → Model Training → Evaluation
     ↓              ↓               ↓                  ↓              ↓
  9,753 records  → 8,615 samples → 1,019 features  → 7 models    → Best: EBM (91.88%)
                   (31倍データ)     (フル活用)        (16並列)      (実用レベル)
```

### Classification System (v0.3)
- **Level Ⅰ (Healthy)**: 1,404 samples (16.3%)
- **Level Ⅱ (Preventive)**: 6,332 samples (73.5%)  
- **Repair-required (III+)**: 879 samples (10.2%) - **44倍増加！**

## 🔧 Technical Features

### 🚀 v0.3 新機能・高速化
- **⚡ 16並列処理**: EBMを25倍高速化（25分→63.8秒）
- **📊 フルデータ学習**: 8,615件の個別記録処理
- **🎯 実行時間トラッキング**: 全モデルの性能監視
- **🔧 最適化パラメータ**: interactions=10, max_bins=64

### 🧠 Machine Learning
- **7 Advanced Models**: EBM, LightGBM, CatBoost, XGBoost, Random Forest
- **Class Imbalance Handling**: Strategic class consolidation and weighting
- **Cross-validation**: 5-fold CV for robust evaluation
- **Interpretable AI**: EBM provides feature importance and decision explanations

### 📝 Text Processing
- **Japanese NLP**: Janome morphological analysis
- **TF-IDF Vectorization**: 1,000-dimensional text features
- **Domain Keywords**: Bridge-specific terminology extraction
- **Multi-modal Features**: Text + numerical + categorical data

### 🛠️ Engineering
- **Automated Pipeline**: End-to-end ML workflow
- **Error Handling**: Robust processing with fallback strategies  
- **Modular Design**: Easily extensible components
- **Performance Monitoring**: Detailed metrics and reporting

## 📁 Project Structure

```
health-ebm-classification/
├── src/
│   ├── main_pipeline.py          # Main execution pipeline (v0.3対応)
│   ├── data_loader.py             # Data loading + フルデータモード
│   ├── feature_engineering.py    # Feature extraction and engineering
│   └── model_trainer.py           # Model training + 16並列処理
├── 1_inspection-dataset/          # Bridge inspection data
├── docs/
│   ├── README_v0-3.md            # 🆕 v0.3 成果まとめ
│   ├── README_v0-2.md            # v0.2 technical documentation  
│   └── QUICK_GUIDE.md            # 5-minute start guide
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 📈 Use Cases

### 🏢 Infrastructure Management
- **Automated Inspection**: Reduce manual assessment time
- **Risk Prioritization**: Identify bridges requiring immediate attention
- **Maintenance Planning**: Data-driven repair scheduling
- **Quality Assurance**: Consistent evaluation standards

### 🔍 Decision Support
- **Explainable Predictions**: Understand why a bridge needs repair
- **Confidence Scoring**: Reliability indicators for each prediction
- **Comparative Analysis**: Benchmark against historical data
- **Expert Validation**: AI recommendations with human oversight

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- 8GB+ RAM (for フルデータ処理)
- **マルチコアCPU推奨** (16並列処理対応)
- Bridge inspection CSV data

### Installation
```bash
pip install pandas numpy scikit-learn
pip install lightgbm xgboost catboost
pip install interpret janome  # For EBM and Japanese text
```

### Data Format
Your CSV files should contain:
- `BridgeID`: Unique bridge identifier
- `HealthLevel`: Current health assessment (Ⅰ, Ⅱ, Ⅲ, Ⅳ, Ⅴ)
- `Diagnosis`: Inspection text description
- `DamageComment`: Detailed damage observations

## 🔮 Roadmap

### ✅ v0.3 (完了) - EBM高速化 & フルデータ学習
- [x] **25倍高速化**: EBM学習時間 25分→63.8秒
- [x] **16並列処理**: CPU最適活用実装
- [x] **フルデータ学習**: 8,615件処理対応
- [x] **実用レベル達成**: Test Accuracy 91.88%
- [x] **包括的ドキュメント**: README_v0-3.md作成

### v1.0 (Production Ready) - 次期リリース
- [ ] REST API implementation
- [ ] Real-time prediction endpoint  
- [ ] Enhanced interpretability dashboard
- [ ] Web application interface

### v1.0 (Production Ready)
- [ ] Web application interface
- [ ] Automated model retraining
- [ ] Integration with inspection databases
- [ ] Multi-language support

### v2.0 (Advanced Features)
- [ ] Image analysis integration
- [ ] Geographic factor modeling
- [ ] Predictive maintenance forecasting
- [ ] Mobile app development

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone with development dependencies
git clone https://github.com/YOUR_USERNAME/health-ebm-classification.git
cd health-ebm-classification

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/health-ebm-classification/issues)
- **Documentation**: See `docs/` folder for detailed guides
- **Questions**: Create a discussion in the repository

## 🏅 Acknowledgments

- **Microsoft InterpretML**: For the Explainable Boosting Machine implementation
- **Japanese NLP Community**: For Janome morphological analyzer
- **Bridge Engineering Domain Experts**: For validation and insights

---

**Built with ❤️ for safer infrastructure**

*Last Updated: October 3, 2025 - v0.3 Release*