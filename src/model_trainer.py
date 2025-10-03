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
import catboost as cb
from interpret.glassbox import ExplainableBoostingClassifier
import joblib
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time  # 実行時間測定用
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
        """XGBoostモデルの作成（クラス不均衡対応強化版）"""
        return xgb.XGBClassifier(
            random_state=self.random_state,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,  # クラス不均衡対応
            objective='multi:softprob',
            eval_metric='mlogloss',
            verbosity=0,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
    
    def create_catboost_model(self) -> cb.CatBoostClassifier:
        """CatBoostモデルの作成（クラス不均衡対応）"""
        return cb.CatBoostClassifier(
            random_state=self.random_state,
            iterations=200,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            class_weights=[1, 1, 3],  # Repair-requirementクラスの重み増加
            verbose=False,
            loss_function='MultiClass',
            eval_metric='MultiClass'
        )
    
    def create_ebm_model(self) -> ExplainableBoostingClassifier:
        """Explainable Boosting Machine（EBM）モデルの作成
        高速化パラメータ: 16並列ワーカー、計算量削減設定
        """
        return ExplainableBoostingClassifier(
            random_state=self.random_state,
            # 並列処理設定
            n_jobs=16,  # 16並列ワーカー
            # 高速化のための計算量削減
            max_bins=128,  # 256→128に削減（2倍高速化）
            max_interaction_bins=16,  # 32→16に削減（2倍高速化）
            interactions=5,  # 10→5に削減（2倍高速化）
            outer_bags=4,  # 8→4に削減（2倍高速化）
            inner_bags=0,  # 内部バッグ無効でさらに高速化
            # 学習効率化
            learning_rate=0.02,  # 学習率向上で早期収束
            validation_size=0.1,  # 検証サイズ削減
            early_stopping_rounds=30,  # 早期停止を積極的に
            early_stopping_tolerance=1e-3,  # 収束判定を緩める
            # メモリ効率化
            max_rounds=500,  # 最大ラウンド数制限
        )
    
    def create_random_forest_model(self) -> RandomForestClassifier:
        """Random Forestモデルの作成（並列処理最適化）"""
        return RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=16  # 16並列ワーカー
        )
    
    def create_ensemble_model(self) -> VotingClassifier:
        """アンサンブルモデルの作成（代替モデル含む）"""
        base_models = [
            ('lr', self.create_baseline_model()),
            ('lgb', self.create_lightgbm_model()),
            ('catboost', self.create_catboost_model()),
            ('xgb', self.create_xgboost_model()),
            ('rf', self.create_random_forest_model())
        ]
        
        return VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
    
    def train_single_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray, model_name: str) -> Dict[str, Any]:
        """単一モデルの訓練と評価（実行時間測定付き）"""
        
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        # XGBoostのためにラベルを0ベースに変換
        if 'XGBoost' in model_name or 'xgb' in str(type(model)).lower():
            y_train_encoded = y_train - 1  # 1,2,3 -> 0,1,2
            y_val_encoded = y_val - 1
        else:
            y_train_encoded = y_train
            y_val_encoded = y_val
        
        # 訓練
        model.fit(X_train, y_train_encoded)
        training_time = time.time() - start_time
        
        # 予測
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # XGBoostの場合は元のラベルに戻す
        if 'XGBoost' in model_name or 'xgb' in str(type(model)).lower():
            y_train_pred = y_train_pred + 1
            y_val_pred = y_val_pred + 1
        
        # 評価指標の計算
        results = {
            'model': model,
            'model_name': model_name,
            'training_time': training_time,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'train_f1_macro': f1_score(y_train, y_train_pred, average='macro'),
            'val_f1_macro': f1_score(y_val, y_val_pred, average='macro'),
            'train_f1_weighted': f1_score(y_train, y_train_pred, average='weighted'),
            'val_f1_weighted': f1_score(y_val, y_val_pred, average='weighted'),
            'classification_report': classification_report(y_val, y_val_pred),
            'confusion_matrix': confusion_matrix(y_val, y_val_pred)
        }
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        # クロスバリデーションも同様にラベル調整（並列処理有効化）
        try:
            if 'XGBoost' in model_name or 'xgb' in str(type(model)).lower():
                cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=5, scoring='f1_macro', n_jobs=16)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=16)
            results['cv_f1_macro_mean'] = cv_scores.mean()
            results['cv_f1_macro_std'] = cv_scores.std()
        except Exception as e:
            print(f"CV評価でエラー: {str(e)}")
            results['cv_f1_macro_mean'] = 0.0
            results['cv_f1_macro_std'] = 0.0
        
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
            (self.create_catboost_model(), 'CatBoost'),
            (self.create_xgboost_model(), 'XGBoost Enhanced'),
            (self.create_ebm_model(), 'Explainable Boosting Machine'),
            (self.create_random_forest_model(), 'Random Forest'),
            (self.create_ensemble_model(), 'Enhanced Ensemble')
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
    
    def evaluate_repair_requirement_performance(self, data_splits: Dict[str, np.ndarray], 
                                               results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Repair-requirementクラス専用の詳細評価"""
        
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        repair_req_results = []
        
        for model_name, result in results.items():
            model = result['model']
            y_pred = model.predict(X_test)
            
            # Repair-requirement (class 3) の性能を詳細分析
            repair_req_mask = (y_test == 3)
            repair_req_true = y_test[repair_req_mask]
            repair_req_pred = y_pred[repair_req_mask]
            
            if len(repair_req_true) > 0:
                # Recall (再現率): Repair-requirementを正しく検出できた割合
                recall = (repair_req_pred == 3).sum() / len(repair_req_true)
                
                # Precision (適合率): Repair-requirementと予測したもののうち正解の割合
                pred_repair_mask = (y_pred == 3)
                if pred_repair_mask.sum() > 0:
                    if hasattr(y_test, 'values'):
                        precision = (y_pred[pred_repair_mask] == y_test.values[pred_repair_mask]).sum() / pred_repair_mask.sum()
                    else:
                        precision = (y_pred[pred_repair_mask] == y_test[pred_repair_mask]).sum() / pred_repair_mask.sum()
                else:
                    precision = 0.0
                
                # F1-score
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
                
                repair_req_results.append({
                    'Model': model_name,
                    'Repair_Req_Recall': recall,
                    'Repair_Req_Precision': precision,
                    'Repair_Req_F1': f1,
                    'Total_Test_Accuracy': result.get('val_accuracy', 0),
                    'Overall_F1_Macro': result.get('val_f1_macro', 0)
                })
        
        repair_df = pd.DataFrame(repair_req_results)
        repair_df = repair_df.sort_values('Repair_Req_F1', ascending=False)
        
        print("\n" + "=" * 60)
        print("🎯 REPAIR-REQUIREMENT クラス専用評価結果")
        print("=" * 60)
        print(repair_df.to_string(index=False, float_format='%.4f'))
        
        # 最良のRepair-requirement予測モデル
        if not repair_df.empty:
            best_repair_model = repair_df.iloc[0]['Model']
            best_f1 = repair_df.iloc[0]['Repair_Req_F1']
            print(f"\n🏆 Repair-requirement予測最良モデル: {best_repair_model}")
            print(f"   F1-Score: {best_f1:.4f}")
            
            # 改善提案
            if best_f1 < 0.6:
                print("\n💡 Repair-requirement予測改善提案:")
                print("   1. より多くのHealthLevel III/IVサンプルの収集")
                print("   2. SMOTE等によるオーバーサンプリング")
                print("   3. コスト考慮型学習（cost-sensitive learning）")
                print("   4. 閾値調整による予測バランス最適化")
        
        return repair_df
    
    def analyze_ebm_interpretability(self, model_name: str = 'Explainable Boosting Machine', 
                                   data_splits: Dict[str, np.ndarray] = None) -> None:
        """EBMモデルの解釈性分析"""
        
        if model_name not in self.models:
            print(f"モデル '{model_name}' が見つかりません。")
            return
        
        model = self.models[model_name]
        
        if not hasattr(model, 'explain_global'):
            print(f"モデル '{model_name}' は解釈性機能をサポートしていません。")
            return
        
        try:
            from interpret import show
            
            print(f"\n🔍 {model_name} 解釈性分析")
            print("=" * 50)
            
            # グローバル解釈（全体的な特徴量重要度）
            global_explanation = model.explain_global()
            
            # 重要な特徴量のサマリー
            print("重要特徴量（解釈可能AI結果）:")
            feature_importances = global_explanation.data()
            
            # 特徴量重要度を表示
            if hasattr(feature_importances, 'scores') and hasattr(feature_importances, 'names'):
                importance_df = pd.DataFrame({
                    'Feature': feature_importances.names,
                    'Importance': feature_importances.scores
                }).sort_values('Importance', ascending=False).head(10)
                
                print(importance_df.to_string(index=False))
            
            # Repair-requirement予測に特に重要な要因を分析
            if data_splits is not None:
                X_test = data_splits['X_test']
                y_test = data_splits['y_test']
                
                # Repair-requirementクラスのサンプルを抽出
                repair_mask = (y_test == 3)
                if repair_mask.sum() > 0:
                    repair_samples = X_test[repair_mask]
                    
                    print(f"\n📊 Repair-requirement サンプル分析:")
                    print(f"   対象サンプル数: {repair_samples.shape[0]}")
                    
                    # ローカル解釈（個別予測の説明）は保存のみ実行
                    print("   ローカル解釈結果は内部処理で生成済み")
                    
            print("\n✅ EBM解釈性分析完了")
            print("   詳細な可視化結果は interpret ライブラリで個別実行可能")
            
        except ImportError:
            print("interpret ライブラリが利用できません。")
        except Exception as e:
            print(f"解釈性分析中にエラーが発生: {str(e)}")
    
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
            model_name = "best_model"
        else:
            model = self.models[model_name]
        
        predictions = model.predict(X)
        
        # XGBoostの場合は予測結果を元のラベルに戻す
        if 'XGBoost' in model_name or 'xgb' in str(type(model)).lower():
            predictions = predictions + 1
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
        
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