# 🚀 HealthLevel分類MVP - Quick Start Guide

## ⚡ 5分で始める橋梁診断AI

### 📋 前提条件
- Python 3.11以上
- 仮想環境(.venv)が設定済み
- CSVデータが`1_inspection-dataset/`フォルダに配置済み

---

## 🔧 Step 1: 環境準備（1分）

```powershell
# 仮想環境アクティブ化
.\.venv\Scripts\Activate.ps1

# 依存関係確認（必要に応じてインストール）
pip install pandas numpy scikit-learn lightgbm xgboost catboost interpret janome
```

## 🚀 Step 2: ワンコマンド実行（3分）

```powershell
# プロジェクトルートから実行
cd src
python main_pipeline.py
```

**これだけで完全なMLパイプラインが実行されます！**

---

## 📊 Step 3: 結果確認（1分）

実行完了後、以下の結果が自動表示されます：

### **最高性能モデル**
```
Best model: Explainable Boosting Machine
Validation F1-macro: 73.61%
Test Accuracy: 84.34%
```

### **クラス別性能**
```
Classification Report:
              precision    recall  f1-score   support
健全(Ⅰ)          0.86      0.86      0.86        29
予防保全(Ⅱ)      0.85      0.92      0.88        48
修繕要(III+)     0.50      0.17      0.25         6
```

---

## 🎯 カスタマイズ実行

### **特定モデルのみ実行**
```python
from model_trainer import ModelTrainer

trainer = ModelTrainer()
# EBMのみ実行
ebm_result = trainer.create_ebm_model(X_train, y_train, X_val, y_val)
```

### **特徴量エンジニアリングのみ**
```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.create_features(data)
print(f"特徴量数: {features.shape[1]}")  # 1027次元
```

### **データ前処理のみ**
```python
from data_loader import BridgeDataLoader

loader = BridgeDataLoader()
data = loader.load_and_preprocess()
print(f"処理後サンプル数: {len(data)}")  # 276サンプル
```

---

## 🔍 重要なファイル

| ファイル | 役割 | 実行例 |
|---------|------|--------|
| `main_pipeline.py` | 完全自動実行 | `python main_pipeline.py` |
| `data_loader.py` | データ読み込み | 個別モジュールとして使用 |
| `feature_engineering.py` | 特徴量生成 | TF-IDF + 専門特徴量 |
| `model_trainer.py` | ML訓練・評価 | 7種類のモデル対応 |

---

## 🚨 よくある問題と解決法

### **Q1: ModuleNotFoundError**
```powershell
# 仮想環境が正しくアクティブ化されているか確認
.\.venv\Scripts\Activate.ps1
pip list | grep -E "(pandas|scikit-learn|lightgbm)"
```

### **Q2: CSVファイルが見つからない**
```
FileNotFoundError: [Errno 2] No such file or directory: '../1_inspection-dataset/v1_InspectionPeriod1.csv'
```
**解決**: `src/`フォルダから実行していることを確認
```powershell
cd src  # このディレクトリから実行
python main_pipeline.py
```

### **Q3: EBMの学習に時間がかかる**
```
Training Explainable Boosting Machine...
```
**正常**: EBMは5-10分程度要します。他モデルは完了しているので待機してください。

### **Q4: NaN値エラー**
```
LogisticRegression does not accept missing values
```
**対応済み**: v0.2でNaN対応モデル（LightGBM, CatBoost, XGBoost, EBM）を優先採用

---

## 📈 成果の読み方

### **モデル選択基準**
- **F1-macro**: 全クラスバランス重視 → **EBM: 73.61%**
- **精度重視**: 全体的な正解率 → **EBM: 86.60%**
- **解釈性重視**: 判定根拠説明 → **EBM推奨**

### **Repair-requirement性能**
```
Precision: 50%  # 修繕要と予測したもののうち正解の割合
Recall: 17%     # 実際の修繕要のうち検出できた割合
F1: 25%         # 総合評価
```
**改善のポイント**: データ追加、アンサンブル手法検討

---

## 🔄 継続改善のヒント

### **性能向上策**
1. **データ拡張**: より多くの修繕要サンプル収集
2. **特徴量追加**: 画像データ、地理的要因
3. **アンサンブル**: 複数モデルの組み合わせ

### **運用改善策**
1. **閾値調整**: Precision/Recall バランス最適化
2. **ドメイン知識統合**: 専門家ルールとの融合
3. **継続学習**: 新データでの定期再訓練

---

## 💡 次のアクション

| 優先度 | アクション | 所要時間 |
|-------|----------|---------|
| 🔥 高 | EBM解釈性分析実行 | 30分 |
| 🔥 高 | Repair-requirement改善実験 | 2時間 |
| 📊 中 | APIエンドポイント作成 | 1日 |
| 🚀 中 | Webアプリ プロトタイプ | 3日 |

---

**🎉 これでHealthLevel分類MVPの実行が完了です！**

**困った時は**: README_v0-2.mdの詳細ドキュメントを参照してください。

---
*Last Updated: 2025年10月3日*