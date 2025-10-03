"""
HealthLevel分類MVPのメインパイプライン
データ読み込みから特徴量作成、モデル訓練、評価まで一連の処理を実行
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_loader import InspectionDataLoader
from feature_engineering import FeatureEngineer
from model_trainer import HealthLevelClassifier

class HealthLevelMVP:
    """HealthLevel分類MVPのメインクラス"""
    
    def __init__(self, data_dir: str, output_dir: str = "../models"):
        """
        Parameters:
        -----------
        data_dir : str
            データディレクトリのパス
        output_dir : str
            出力ディレクトリのパス
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_loader = InspectionDataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        self.classifier = HealthLevelClassifier()
        
        self.raw_data = None
        self.processed_data = None
        self.aggregated_data = None
        self.features = None
        self.target = None
        self.results = {}
        
    def step1_data_understanding(self):
        """ステップ1: データ理解・収集"""
        print("=" * 50)
        print("STEP 1: データ理解・収集")
        print("=" * 50)
        
        # データ読み込み
        self.raw_data = self.data_loader.load_data()
        
        # 基本情報の表示
        info = self.data_loader.get_basic_info()
        print("\n=== データ基本情報 ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # データの先頭を表示
        print("\n=== データサンプル ===")
        print(self.raw_data.head())
        
        # HealthLevel分布の可視化
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        health_counts = self.raw_data['HealthLevel'].value_counts()
        plt.pie(health_counts.values, labels=health_counts.index, autopct='%1.1f%%')
        plt.title('HealthLevel Distribution')
        
        plt.subplot(1, 2, 2)
        sns.countplot(data=self.raw_data, x='HealthLevel')
        plt.title('HealthLevel Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'health_level_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return info
    
    def step2_data_preprocessing(self):
        """ステップ2: 前処理（Re-Cleansing）"""
        print("\n" + "=" * 50)
        print("STEP 2: 前処理（Re-Cleansing）")
        print("=" * 50)
        
        # 基本前処理
        self.processed_data = self.data_loader.basic_preprocessing()
        
        # 集約特徴量の作成
        self.aggregated_data = self.data_loader.create_aggregated_features()
        
        print(f"\n処理後データ形状: {self.aggregated_data.shape}")
        print(f"集約後のHealthLevel分布:")
        print(self.aggregated_data['HealthLevel'].value_counts())
        
        # 欠損値の確認
        missing_info = self.aggregated_data.isnull().sum()
        print(f"\n欠損値情報:")
        print(missing_info[missing_info > 0])
        
        return self.aggregated_data
    
    def step3_data_split(self):
        """ステップ3: データ分割（Split）"""
        print("\n" + "=" * 50)
        print("STEP 3: データ分割（Split）")
        print("=" * 50)
        
        # HealthLevelのエンコーディング（III以上をRepair-requirementクラスに統合）
        def encode_health_level(level):
            if level == 'Ⅰ':
                return 1
            elif level == 'Ⅱ':
                return 2
            elif level in ['Ⅲ', 'Ⅳ', 'Ⅴ']:
                return 3  # Repair-requirement クラス
            else:
                return None  # Nやその他の値は除外
        
        # 集約データにhealth_level_encodedがない場合は作成
        if 'health_level_encoded' not in self.aggregated_data.columns:
            self.aggregated_data['health_level_encoded'] = self.aggregated_data['HealthLevel'].apply(encode_health_level)
            # Nレベルを除外
            self.aggregated_data = self.aggregated_data[self.aggregated_data['health_level_encoded'].notna()].copy()
        
        self.aggregated_data['target'] = self.aggregated_data['health_level_encoded']
        
        # 特徴量とターゲットの分離
        self.target = self.aggregated_data['target'].values
        
        print(f"ターゲット分布:")
        target_counts = pd.Series(self.target).value_counts().sort_index()
        reverse_mapping = {1: 'Ⅰ', 2: 'Ⅱ', 3: 'Repair-requirement'}
        for target, count in target_counts.items():
            health_level = reverse_mapping.get(target, f'Unknown-{target}')
            print(f"  {health_level} (Level {target}): {count} samples ({count/len(self.target)*100:.1f}%)")
        
        return self.target
    
    def step4_feature_engineering(self):
        """ステップ4: 特徴量エンジニアリング"""
        print("\n" + "=" * 50)
        print("STEP 4: 特徴量エンジニアリング")
        print("=" * 50)
        
        # 特徴量の作成
        self.features, feature_names = self.feature_engineer.create_all_features(
            self.aggregated_data,
            text_columns=['diagnosis_text', 'damage_comment_text'],
            categorical_columns=['BridgeName', 'inspection_year', 'inspection_month'],
            numerical_columns=['diagnosis_count', 'damage_count', 'damage_rank_mean', 
                             'damage_rank_max', 'crack_width_mean', 'crack_width_max', 
                             'area_sum', 'area_max'],
            fit=True
        )
        
        self.classifier.feature_names = feature_names
        
        print(f"特徴量数: {self.features.shape[1]}")
        print(f"サンプル数: {self.features.shape[0]}")
        print(f"特徴量の種類:")
        
        # 特徴量タイプの集計
        feature_types = {}
        for name in feature_names:
            if name.startswith('tfidf_'):
                feature_types['TF-IDF'] = feature_types.get('TF-IDF', 0) + 1
            elif name.startswith('keyword_'):
                feature_types['Keyword'] = feature_types.get('Keyword', 0) + 1
            elif name.startswith('has_'):
                feature_types['Binary'] = feature_types.get('Binary', 0) + 1
            elif name.startswith('cat_'):
                feature_types['Categorical'] = feature_types.get('Categorical', 0) + 1
            elif name.startswith('num_'):
                feature_types['Numerical'] = feature_types.get('Numerical', 0) + 1
            else:
                feature_types['Other'] = feature_types.get('Other', 0) + 1
        
        for ftype, count in feature_types.items():
            print(f"  {ftype}: {count}")
        
        return self.features, feature_names
    
    def step5_model_building(self):
        """ステップ5: モデル構築＆学習"""
        print("\n" + "=" * 50)
        print("STEP 5: モデル構築＆学習")
        print("=" * 50)
        
        # データ分割
        data_splits = self.classifier.prepare_data(self.features, self.target)
        
        print(f"訓練データ: {len(data_splits['X_train'])} samples")
        print(f"検証データ: {len(data_splits['X_val'])} samples")
        print(f"テストデータ: {len(data_splits['X_test'])} samples")
        
        # 全モデルの訓練
        self.results = self.classifier.train_all_models(data_splits)
        
        # 最良モデルの選択
        best_model_name = self.classifier.select_best_model(self.results)
        
        return data_splits, best_model_name
    
    def step6_evaluation(self, data_splits, best_model_name):
        """ステップ6: 評価（Evaluate）"""
        print("\n" + "=" * 50)
        print("STEP 6: 評価（Evaluate）")
        print("=" * 50)
        
        # テストデータでの最終評価
        test_results = self.classifier.evaluate_on_test(data_splits)
        
        # 混同行列の可視化
        cm = test_results['confusion_matrix']
        class_names = ['Ⅰ', 'Ⅱ', 'Repair-requirement'][:cm.shape[0]]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 特徴量重要度の表示
        feature_importance = self.classifier.get_feature_importance(top_k=15)
        if not feature_importance.empty:
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance, y='feature', x='importance')
            plt.title('Feature Importance (Top 15)')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\n=== Top 15 重要特徴量 ===")
            print(feature_importance.to_string(index=False))
        
        # モデル比較結果の保存
        comparison_df = pd.DataFrame({
            'Model': [result['model_name'] for result in self.results.values()],
            'Validation Accuracy': [result['val_accuracy'] for result in self.results.values()],
            'Validation F1 (macro)': [result['val_f1_macro'] for result in self.results.values()],
            'CV F1 (macro)': [result['cv_f1_macro_mean'] for result in self.results.values()]
        })
        
        print("\n=== モデル比較結果 ===")
        print(comparison_df.to_string(index=False))
        
        comparison_df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        
        # 🎯 Repair-requirement専用評価
        repair_performance = self.classifier.evaluate_repair_requirement_performance(data_splits, self.results)
        repair_performance.to_csv(self.output_dir / 'repair_requirement_performance.csv', index=False)
        
        # 🔍 EBM解釈性分析
        self.classifier.analyze_ebm_interpretability('Explainable Boosting Machine', data_splits)
        
        return test_results, feature_importance
    
    def step7_deploy(self, best_model_name):
        """ステップ7: デプロイ＆可視化（Use it!）"""
        print("\n" + "=" * 50)
        print("STEP 7: デプロイ＆可視化（Use it!）")
        print("=" * 50)
        
        # モデルの保存
        model_path = self.classifier.save_model(best_model_name, 
                                               self.output_dir / f"{best_model_name.lower().replace(' ', '_')}.joblib")
        
        # 特徴量エンジニアリングパイプラインの保存
        import joblib
        feature_pipeline_path = self.output_dir / "feature_engineer.joblib"
        joblib.dump(self.feature_engineer, feature_pipeline_path)
        print(f"Feature engineering pipeline saved to: {feature_pipeline_path}")
        
        # 予測関数のサンプル作成
        self.create_prediction_script()
        
        print(f"\n=== デプロイメント完了 ===")
        print(f"モデルファイル: {model_path}")
        print(f"特徴量パイプライン: {feature_pipeline_path}")
        print(f"予測スクリプト: {self.output_dir / 'predict.py'}")
        
        return model_path
    
    def create_prediction_script(self):
        """予測用スクリプトの作成"""
        prediction_script = '''"""
HealthLevel予測スクリプト
新しい橋梁診断データに対してHealthLevelを予測する
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class HealthLevelPredictor:
    """HealthLevel予測器"""
    
    def __init__(self, model_path: str, feature_engineer_path: str):
        """
        Parameters:
        -----------
        model_path : str
            訓練済みモデルのパス
        feature_engineer_path : str
            特徴量エンジニアリングパイプラインのパス
        """
        self.model = joblib.load(model_path)
        self.feature_engineer = joblib.load(feature_engineer_path)
        self.label_mapping = {1: 'Ⅰ', 2: 'Ⅱ', 3: 'Repair-requirement'}
        
    def predict(self, diagnosis_text: str, damage_comment: str = "", 
                bridge_name: str = "Unknown", inspection_date: str = "2024-01-01",
                damage_rank_mean: float = 2.0, damage_count: int = 1) -> dict:
        """
        HealthLevelを予測
        
        Parameters:
        -----------
        diagnosis_text : str
            診断テキスト
        damage_comment : str
            損傷コメント
        bridge_name : str
            橋梁名
        inspection_date : str
            点検日（YYYY-MM-DD形式）
        damage_rank_mean : float
            平均損傷ランク
        damage_count : int
            損傷数
            
        Returns:
        --------
        dict
            予測結果
        """
        
        # 入力データの作成
        input_data = pd.DataFrame({
            'diagnosis_text': [diagnosis_text],
            'damage_comment_text': [damage_comment],
            'BridgeName': [bridge_name],
            'InspectionYMD': [inspection_date],
            'damage_rank_mean': [damage_rank_mean],
            'damage_count': [damage_count],
            'diagnosis_count': [1],
            'damage_rank_max': [damage_rank_mean],
            'crack_width_mean': [np.nan],
            'crack_width_max': [np.nan],
            'area_sum': [np.nan],
            'area_max': [np.nan]
        })
        
        # 特徴量の作成
        features, _ = self.feature_engineer.create_all_features(
            input_data,
            text_columns=['diagnosis_text', 'damage_comment_text'],
            categorical_columns=['BridgeName', 'inspection_year', 'inspection_month'],
            numerical_columns=['diagnosis_count', 'damage_count', 'damage_rank_mean', 
                             'damage_rank_max', 'crack_width_mean', 'crack_width_max', 
                             'area_sum', 'area_max'],
            fit=False
        )
        
        # 予測実行
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0] if hasattr(self.model, 'predict_proba') else None
        
        # 結果の整理
        result = {
            'predicted_health_level': self.label_mapping[prediction],
            'predicted_health_level_numeric': prediction,
            'confidence': float(probabilities[prediction-1]) if probabilities is not None else None,
            'all_probabilities': {
                self.label_mapping[i+1]: float(prob) 
                for i, prob in enumerate(probabilities)
            } if probabilities is not None else None
        }
        
        return result

def main():
    """予測スクリプトのテスト実行"""
    
    # 予測器の初期化
    predictor = HealthLevelPredictor(
        model_path="best_model.joblib",
        feature_engineer_path="feature_engineer.joblib"
    )
    
    # サンプル予測
    result = predictor.predict(
        diagnosis_text="主桁に鉄筋露出が見られる。間詰め床版に遊離石灰が見られる。",
        damage_comment="鉄筋が露出しており、鉄筋が腐食している。",
        bridge_name="山田橋",
        inspection_date="2024-03-01",
        damage_rank_mean=3.0,
        damage_count=2
    )
    
    print("=== 予測結果 ===")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
'''
        
        with open(self.output_dir / 'predict.py', 'w', encoding='utf-8') as f:
            f.write(prediction_script)
    
    def run_full_pipeline(self):
        """全パイプラインの実行"""
        print("🚀 HealthLevel分類MVP パイプライン開始")
        print("=" * 60)
        
        try:
            # ステップ1: データ理解
            info = self.step1_data_understanding()
            
            # ステップ2: 前処理
            processed_data = self.step2_data_preprocessing()
            
            # ステップ3: データ分割
            target = self.step3_data_split()
            
            # ステップ4: 特徴量エンジニアリング
            features, feature_names = self.step4_feature_engineering()
            
            # ステップ5: モデル構築
            data_splits, best_model_name = self.step5_model_building()
            
            # ステップ6: 評価
            test_results, feature_importance = self.step6_evaluation(data_splits, best_model_name)
            
            # ステップ7: デプロイ
            model_path = self.step7_deploy(best_model_name)
            
            print("\n🎉 パイプライン完了！")
            print("=" * 60)
            print(f"最終テスト精度: {test_results['test_accuracy']:.4f}")
            print(f"最終テストF1スコア: {test_results['test_f1_macro']:.4f}")
            print(f"最良モデル: {best_model_name}")
            
            return {
                'best_model_name': best_model_name,
                'test_accuracy': test_results['test_accuracy'],
                'test_f1_macro': test_results['test_f1_macro'],
                'model_path': model_path,
                'feature_count': len(feature_names)
            }
            
        except Exception as e:
            print(f"❌ エラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """メイン実行"""
    
    # MVPパイプラインの実行
    mvp = HealthLevelMVP(
        data_dir="../1_inspection-dataset",
        output_dir="../models"
    )
    
    results = mvp.run_full_pipeline()
    
    if results:
        print("\n📊 最終結果サマリー:")
        for key, value in results.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()