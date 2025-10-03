# 🏗️ Health EBM Classification

**Bridge Health Level Classification using Explainable Boosting Machine**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-MVP%20v0.2-orange.svg)]()

## 🎯 Overview

An AI-powered bridge health classification system that automatically categorizes bridge inspection reports into health levels using machine learning. The system leverages **Explainable Boosting Machine (EBM)** to achieve high accuracy while maintaining interpretability.

### 🏆 Key Achievements (v0.2)
- **73.61% F1-macro score** on validation set
- **86.60% accuracy** with interpretable predictions
- **Repair-requirement detection** for proactive maintenance
- **Japanese text processing** with domain-specific features

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

## 📊 Performance Results

| Model | F1-Macro | Accuracy | Interpretability |
|-------|----------|----------|------------------|
| **🥇 EBM** | **73.61%** | **86.60%** | ✅ High |
| LightGBM | 70.34% | 83.51% | ❌ Low |
| CatBoost | 66.06% | 84.54% | ❌ Low |
| XGBoost | 65.70% | 84.54% | ❌ Low |

## 🏗️ Architecture

### Data Pipeline
```
Raw CSV Data → Preprocessing → Feature Engineering → Model Training → Evaluation
     ↓              ↓               ↓                  ↓              ↓
  9,753 records  → 276 samples  → 1,027 features  → 7 models    → Best: EBM
```

### Classification System
- **Level Ⅰ (Healthy)**: 96 samples (34.8%)
- **Level Ⅱ (Preventive)**: 160 samples (58.0%)  
- **Repair-required (III+)**: 20 samples (7.2%)

## 🔧 Technical Features

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
│   ├── main_pipeline.py          # Main execution pipeline
│   ├── data_loader.py             # Data loading and preprocessing
│   ├── feature_engineering.py    # Feature extraction and engineering
│   └── model_trainer.py           # Model training and evaluation
├── 1_inspection-dataset/          # Bridge inspection data
├── docs/
│   ├── README_v0-2.md            # Detailed technical documentation
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
- 8GB+ RAM (for EBM training)
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

### v0.3 (Next Release)
- [ ] REST API implementation
- [ ] Real-time prediction endpoint
- [ ] Enhanced interpretability dashboard
- [ ] Improved Repair-requirement recall (target: 50%+)

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

*Last Updated: October 3, 2025*