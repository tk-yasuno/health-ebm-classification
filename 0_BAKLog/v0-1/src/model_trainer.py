"""
モデル訓練モジュール
ベースラインから高性能モデルまでの訓練・評価を行う
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
import joblib
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class HealthLevelClassifier:
    """橋梁健全度レベル分類器"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.feature_names = []
        self.label_mapping = {'Ⅰ': 1, 'Ⅱ': 2, 'Repair-requirement': 3}
        self.reverse_label_mapping = {1: 'Ⅰ', 2: 'Ⅱ', 3: 'Repair-requirement'}
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.3, val_size: float = 0.5) -> Dict[str, np.ndarray]:
        """データの分割（Train/Validation/Test）"""
        
        # クラス数の確認（サンプル数が少ないクラスを除外するか確認）
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        
        print(f"最小クラスのサンプル数: {min_samples}")
        
        # サンプル数が極端に少ないクラス（2未満）がある場合は単純分割
        if min_samples < 2:
            print("警告: サンプル数が極端に少ないクラスがあるため、単純分割を使用します")
            
            # まずTrain+ValidationとTestに分割
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
            # Train+ValidationをTrainとValidationに分割
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=self.random_state
            )
        else:
            # まずTrain+ValidationとTestに分割
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, 
                stratify=y
            )
            
            # Train+ValidationをTrainとValidationに分割
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=self.random_state,
                stratify=y_temp
            )
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
    
    def create_baseline_model(self) -> LogisticRegression:
        """ベースラインモデル（ロジスティック回帰）の作成"""
        return LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
    
    def create_lightgbm_model(self) -> lgb.LGBMClassifier:
        """LightGBMモデルの作成"""
        return lgb.LGBMClassifier(
            random_state=self.random_state,
            class_weight='balanced',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multiclass',
            verbose=-1
        )
    
    def create_xgboost_model(self) -> xgb.XGBClassifier:
        """XGBoostモデルの作成"""
        return xgb.XGBClassifier(
            random_state=self.random_state,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            verbosity=0
        )
    
    def create_random_forest_model(self) -> RandomForestClassifier:
        """Random Forestモデルの作成"""
        return RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced'
        )
    
    def create_ensemble_model(self) -> VotingClassifier:
        """アンサンブルモデルの作成"""
        base_models = [
            ('lr', self.create_baseline_model()),
            ('lgb', self.create_lightgbm_model()),
            ('xgb', self.create_xgboost_model()),
            ('rf', self.create_random_forest_model())
        ]
        
        return VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
    
    def train_single_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray, model_name: str) -> Dict[str, Any]:
        """単一モデルの訓練と評価"""
        
        print(f"\nTraining {model_name}...")
        
        # 訓練
        model.fit(X_train, y_train)
        
        # 予測
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # 評価指標の計算
        results = {
            'model': model,
            'model_name': model_name,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'train_f1_macro': f1_score(y_train, y_train_pred, average='macro'),
            'val_f1_macro': f1_score(y_val, y_val_pred, average='macro'),
            'train_f1_weighted': f1_score(y_train, y_train_pred, average='weighted'),
            'val_f1_weighted': f1_score(y_val, y_val_pred, average='weighted'),
            'classification_report': classification_report(y_val, y_val_pred),
            'confusion_matrix': confusion_matrix(y_val, y_val_pred)
        }
        
        # クロスバリデーション
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
        results['cv_f1_macro_mean'] = cv_scores.mean()
        results['cv_f1_macro_std'] = cv_scores.std()
        
        print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"Validation F1 (macro): {results['val_f1_macro']:.4f}")
        print(f"CV F1 (macro): {results['cv_f1_macro_mean']:.4f} ± {results['cv_f1_macro_std']:.4f}")
        
        return results
    
    def train_all_models(self, data_splits: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """全モデルの訓練と比較"""
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        
        models_to_train = [
            (self.create_baseline_model(), 'Logistic Regression'),
            (self.create_lightgbm_model(), 'LightGBM'),
            (self.create_xgboost_model(), 'XGBoost'),
            (self.create_random_forest_model(), 'Random Forest'),
            (self.create_ensemble_model(), 'Ensemble')
        ]
        
        results = {}
        for model, name in models_to_train:
            try:
                result = self.train_single_model(model, X_train, y_train, X_val, y_val, name)
                results[name] = result
                self.models[name] = model
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        return results
    
    def select_best_model(self, results: Dict[str, Dict[str, Any]], 
                         metric: str = 'val_f1_macro') -> str:
        """最良モデルの選択"""
        
        best_score = -1
        best_model_name = None
        
        print(f"\n=== Model Comparison (by {metric}) ===")
        for name, result in results.items():
            score = result[metric]
            print(f"{name}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = name
        
        print(f"\nBest model: {best_model_name} (score: {best_score:.4f})")
        self.best_model = self.models[best_model_name]
        
        return best_model_name
    
    def evaluate_on_test(self, data_splits: Dict[str, np.ndarray], 
                        model_name: Optional[str] = None) -> Dict[str, Any]:
        """テストデータでの最終評価"""
        
        if model_name is None:
            model = self.best_model
            model_name = "Best Model"
        else:
            model = self.models[model_name]
        
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        test_results = {
            'model_name': model_name,
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1_macro': f1_score(y_test, y_pred, average='macro'),
            'test_f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"\n=== Test Results for {model_name} ===")
        print(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
        print(f"Test F1 (macro): {test_results['test_f1_macro']:.4f}")
        print(f"Test F1 (weighted): {test_results['test_f1_weighted']:.4f}")
        print(f"\nClassification Report:\n{test_results['classification_report']}")
        
        return test_results
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, 
                            class_names: List[str] = None) -> None:
        """混同行列の可視化"""
        
        plt.figure(figsize=(8, 6))
        
        if class_names is None:
            class_names = [self.reverse_label_mapping.get(i+1, f'Class {i+1}') 
                          for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, model_name: str = None, top_k: int = 20) -> pd.DataFrame:
        """特徴量重要度の取得"""
        
        if model_name is None:
            model = self.best_model
            model_name = "Best Model"
        else:
            model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # ロジスティック回帰の場合は係数の絶対値の平均
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            print(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(importances))],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_k)
    
    def save_model(self, model_name: str = None, filepath: str = None) -> str:
        """モデルの保存"""
        
        if model_name is None:
            model = self.best_model
            model_name = "best_model"
        else:
            model = self.models[model_name]
        
        if filepath is None:
            filepath = f"../models/{model_name.lower().replace(' ', '_')}.joblib"
        
        joblib.dump(model, filepath)
        print(f"Model saved to: {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> Any:
        """モデルの読み込み"""
        
        model = joblib.load(filepath)
        print(f"Model loaded from: {filepath}")
        
        return model
    
    def predict(self, X: np.ndarray, model_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """新しいデータに対する予測"""
        
        if model_name is None:
            model = self.best_model
        else:
            model = self.models[model_name]
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        return predictions, probabilities

def main():
    """モデル訓練のテスト実行"""
    
    # サンプルデータの生成
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(1, 4, n_samples)  # HealthLevel 1-3
    
    # 分類器の初期化
    classifier = HealthLevelClassifier()
    
    # データ分割
    data_splits = classifier.prepare_data(X, y)
    
    print(f"Training set size: {len(data_splits['X_train'])}")
    print(f"Validation set size: {len(data_splits['X_val'])}")
    print(f"Test set size: {len(data_splits['X_test'])}")
    
    # 全モデルの訓練
    results = classifier.train_all_models(data_splits)
    
    # 最良モデルの選択
    best_model_name = classifier.select_best_model(results)
    
    # テストデータでの評価
    test_results = classifier.evaluate_on_test(data_splits)
    
    # 混同行列の表示
    classifier.plot_confusion_matrix(
        test_results['confusion_matrix'], 
        best_model_name
    )

if __name__ == "__main__":
    main()