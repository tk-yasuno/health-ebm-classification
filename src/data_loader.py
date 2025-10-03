"""
データローダーモジュール
橋梁点検データの読み込みと基本的な前処理を行う
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
import re
import warnings
warnings.filterwarnings('ignore')

class InspectionDataLoader:
    """橋梁点検データの読み込みと前処理を行うクラス"""
    
    def __init__(self, data_dir: str):
        """
        Parameters:
        -----------
        data_dir : str
            データディレクトリのパス
        """
        self.data_dir = Path(data_dir)
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self, file_patterns: List[str] = None) -> pd.DataFrame:
        """
        CSVファイルを読み込み、結合する
        
        Parameters:
        -----------
        file_patterns : List[str], optional
            読み込むファイルのパターン。Noneの場合は全CSVファイル
            
        Returns:
        --------
        pd.DataFrame
            結合されたデータフレーム
        """
        if file_patterns is None:
            csv_files = list(self.data_dir.glob("*.csv"))
        else:
            csv_files = []
            for pattern in file_patterns:
                csv_files.extend(list(self.data_dir.glob(pattern)))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        dfs = []
        for file_path in csv_files:
            print(f"Loading: {file_path.name}")
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            df['source_file'] = file_path.name
            dfs.append(df)
        
        self.raw_data = pd.concat(dfs, ignore_index=True)
        print(f"Total records loaded: {len(self.raw_data)}")
        
        return self.raw_data
    
    def get_basic_info(self) -> dict:
        """データの基本情報を取得"""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        info = {
            'total_records': len(self.raw_data),
            'columns': list(self.raw_data.columns),
            'health_level_distribution': self.raw_data['HealthLevel'].value_counts().to_dict(),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'unique_bridges': self.raw_data['BridgeID'].nunique(),
            'date_range': {
                'min': self.raw_data['InspectionYMD'].min(),
                'max': self.raw_data['InspectionYMD'].max()
            }
        }
        
        return info
    
    def basic_preprocessing(self) -> pd.DataFrame:
        """基本的な前処理を実行"""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df = self.raw_data.copy()
        
        # 日付型に変換
        df['InspectionYMD'] = pd.to_datetime(df['InspectionYMD'])
        
        # 年月の抽出
        df['inspection_year'] = df['InspectionYMD'].dt.year
        df['inspection_month'] = df['InspectionYMD'].dt.month
        
        # DamageRankのカテゴリ化
        df['damage_rank_encoded'] = df['DamageRank'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
        
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
        
        df['health_level_encoded'] = df['HealthLevel'].apply(encode_health_level)
        
        # 'N'レベル（評価対象外）を除外
        df = df[df['health_level_encoded'].notna()].copy()
        
        # テキストの基本的なクリーニング
        df['diagnosis_cleaned'] = df['Diagnosis'].apply(self._clean_text)
        df['damage_comment_cleaned'] = df['DamageComment'].apply(self._clean_text)
        
        # 数値的特徴量の抽出（ひび割れ幅、面積など）
        df['crack_width'] = df['DamageComment'].apply(self._extract_crack_width)
        df['area_measurement'] = df['DamageComment'].apply(self._extract_area)
        
        self.processed_data = df
        print(f"Preprocessing completed. Shape: {df.shape}")
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """テキストのクリーニング"""
        if pd.isna(text):
            return ""
        
        # 全角数字を半角に変換
        text = text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        
        # 全角英字を半角に変換
        text = text.translate(str.maketrans('ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ',
                                             'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'))
        
        # 改行文字の除去
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # 連続する空白の正規化
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_crack_width(self, text: str) -> float:
        """ひび割れ幅の数値を抽出（mm単位）"""
        if pd.isna(text):
            return np.nan
        
        # ひび割れ幅のパターンを検索
        patterns = [
            r'ひびわれ幅[\s]*([0-9.]+)[\s]*mm',
            r'ひび割れ幅[\s]*([0-9.]+)[\s]*mm',
            r'幅[\s]*([0-9.]+)[\s]*mm',
            r'([0-9.]+)[\s]*mm'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return np.nan
    
    def _extract_area(self, text: str) -> float:
        """面積の数値を抽出（m×mのパターン）"""
        if pd.isna(text):
            return np.nan
        
        # 面積のパターンを検索 (例: 0.8m×0.2m)
        pattern = r'([0-9.]+)m×([0-9.]+)m'
        match = re.search(pattern, text)
        
        if match:
            try:
                width = float(match.group(1))
                height = float(match.group(2))
                return width * height
            except ValueError:
                return np.nan
        
        return np.nan
    
    def create_aggregated_features(self, use_full_data: bool = False) -> pd.DataFrame:
        """橋梁レベルでの集約特徴量を作成
        
        Parameters:
        -----------
        use_full_data : bool, default=False
            Trueの場合、個別レコードレベルでの特徴量を使用（9753件）
            Falseの場合、橋梁別集約特徴量を使用（276件）
        """
        if self.processed_data is None:
            raise ValueError("Data not preprocessed. Call basic_preprocessing() first.")
        
        if use_full_data:
            print("🚀 フルデータモード: 個別レコードレベル（9753件）で学習")
            # 個別レコードレベルでの特徴量
            full_data = self.processed_data.copy()
            
            # HealthLevelのエンコーディング
            def encode_health_level(level):
                if level == 'Ⅰ':
                    return 1
                elif level == 'Ⅱ':
                    return 2
                elif level in ['Ⅲ', 'Ⅳ', 'Ⅴ']:
                    return 3  # Repair-requirement クラス
                else:
                    return None
            
            full_data['health_level_encoded'] = full_data['HealthLevel'].apply(encode_health_level)
            full_data = full_data[full_data['health_level_encoded'].notna()].copy()
            
            # テキスト特徴量の準備
            full_data['combined_text'] = (
                full_data['diagnosis_cleaned'].fillna('') + ' ' + 
                full_data['damage_comment_cleaned'].fillna('')
            ).str.strip()
            
            # 基本的な数値特徴量
            feature_columns = [
                'BridgeID', 'health_level_encoded', 'DamageRank', 
                'crack_width', 'area_measurement', 'combined_text'
            ]
            
            # 数値特徴量の欠損値処理
            full_data['crack_width'] = full_data['crack_width'].fillna(0)
            full_data['area_measurement'] = full_data['area_measurement'].fillna(0)
            full_data['DamageRank'] = pd.to_numeric(full_data['DamageRank'], errors='coerce').fillna(1)
            
            result_data = full_data[feature_columns].copy()
            print(f"フルデータ形状: {result_data.shape}")
            print(f"HealthLevel分布:")
            print(full_data['HealthLevel'].value_counts())
            
            return result_data
        
        else:
            print("📊 集約モード: 橋梁別集約（276件）で学習")
            # 橋梁×診断日レベルでの集約（元のロジック）
            agg_features = self.processed_data.groupby(['BridgeID', 'BridgeName', 'InspectionYMD', 'HealthLevel']).agg({
                'DiagnosisID': 'nunique',  # 診断項目数
                'DamageID': 'nunique',     # 損傷数
                'damage_rank_encoded': ['mean', 'max'],  # 損傷ランクの平均・最大
                'crack_width': ['mean', 'max', 'count'],  # ひび割れ幅の統計
                'area_measurement': ['sum', 'max'],       # 面積の統計
                'Diagnosis': lambda x: ' '.join(x.dropna().astype(str).unique()),  # 診断テキストの結合
                'DamageComment': lambda x: ' '.join(x.dropna().astype(str).unique())  # 損傷コメントの結合
            }).reset_index()
            
            # カラム名の整理
            agg_features.columns = [
                'BridgeID', 'BridgeName', 'InspectionYMD', 'HealthLevel',
                'diagnosis_count', 'damage_count',
                'damage_rank_mean', 'damage_rank_max',
                'crack_width_mean', 'crack_width_max', 'crack_width_count',
                'area_sum', 'area_max',
                'diagnosis_text', 'damage_comment_text'
            ]
            
            # テキストクリーニング
            agg_features['diagnosis_text'] = agg_features['diagnosis_text'].apply(self._clean_text)
            agg_features['damage_comment_text'] = agg_features['damage_comment_text'].apply(self._clean_text)
            
            return agg_features

def main():
    """データローダーのテスト実行"""
    loader = InspectionDataLoader("../1_inspection-dataset")
    
    # データ読み込み
    raw_data = loader.load_data()
    print("\n=== Basic Info ===")
    info = loader.get_basic_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 前処理実行
    print("\n=== Preprocessing ===")
    processed_data = loader.basic_preprocessing()
    
    # 集約特徴量作成
    print("\n=== Aggregated Features ===")
    agg_data = loader.create_aggregated_features()
    print(f"Aggregated data shape: {agg_data.shape}")
    print(agg_data.head())

if __name__ == "__main__":
    main()