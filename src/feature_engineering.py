"""
特徴量エンジニアリングモジュール
テキスト特徴量、カテゴリ特徴量、数値特徴量の生成を行う
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from janome.tokenizer import Tokenizer
import re
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """特徴量エンジニアリングを行うクラス"""
    
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tfidf_vectorizer = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # ストップワードの定義
        self.stop_words = {
            'が', 'を', 'に', 'は', 'で', 'と', 'の', 'から', 'まで', 'より', 'も',
            'た', 'だ', 'である', 'です', 'ます', 'した', 'する', 'される', 'れる',
            'ある', 'いる', 'なる', 'こと', 'もの', 'ため', 'など', 'として',
            'による', 'により', 'について', 'において', 'に関して', 'に対して'
        }
        
        # 橋梁関連の重要キーワード
        self.important_keywords = {
            '損傷': ['ひび', 'ひびわれ', 'ひび割れ', 'クラック', '亀裂'],
            '腐食': ['腐食', '錆', 'さび', '鉄筋露出', '露出'],
            '劣化': ['劣化', '中性化', '浮き', '剥離', '剥落'],
            '漏水': ['漏水', '滞水', '遊離石灰', '白華'],
            '変形': ['変形', 'たわみ', '沈下', '移動'],
            '部材': ['主桁', '床版', '橋脚', '橋台', '支承', '伸縮装置', '防護柵', '舗装']
        }
        
    def tokenize_japanese(self, text: str) -> List[str]:
        """日本語テキストの形態素解析"""
        if pd.isna(text) or text == "":
            return []
        
        tokens = []
        for token in self.tokenizer.tokenize(text, wakati=True):
            # 長さ2文字以上、ストップワード除外
            if len(token) >= 2 and token not in self.stop_words:
                # 英数字のみは除外
                if not re.match(r'^[a-zA-Z0-9]+$', token):
                    tokens.append(token)
        
        return tokens
    
    def extract_keyword_features(self, text: str) -> Dict[str, int]:
        """重要キーワードの存在をチェック"""
        if pd.isna(text):
            text = ""
        
        text_lower = text.lower()
        features = {}
        
        for category, keywords in self.important_keywords.items():
            count = 0
            for keyword in keywords:
                count += text_lower.count(keyword)
            features[f'keyword_{category}'] = count
            features[f'has_{category}'] = 1 if count > 0 else 0
        
        return features
    
    def extract_numerical_features(self, text: str) -> Dict[str, float]:
        """テキストから数値特徴量を抽出"""
        if pd.isna(text):
            return {
                'crack_width_mm': np.nan,
                'area_m2': np.nan,
                'length_m': np.nan,
                'has_measurement': 0
            }
        
        features = {
            'crack_width_mm': np.nan,
            'area_m2': np.nan, 
            'length_m': np.nan,
            'has_measurement': 0
        }
        
        # ひび割れ幅 (mm)
        crack_patterns = [
            r'ひびわれ幅[\s]*([0-9.]+)[\s]*mm',
            r'ひび割れ幅[\s]*([0-9.]+)[\s]*mm',
            r'幅[\s]*([0-9.]+)[\s]*mm'
        ]
        
        for pattern in crack_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    features['crack_width_mm'] = float(match.group(1))
                    features['has_measurement'] = 1
                    break
                except ValueError:
                    continue
        
        # 面積 (m×m)
        area_pattern = r'([0-9.]+)m×([0-9.]+)m'
        match = re.search(area_pattern, text)
        if match:
            try:
                width = float(match.group(1))
                height = float(match.group(2))
                features['area_m2'] = width * height
                features['has_measurement'] = 1
            except ValueError:
                pass
        
        # 長さ (m)
        length_pattern = r'([0-9.]+)m'
        matches = re.findall(length_pattern, text)
        if matches:
            try:
                # 最大の長さを取得
                lengths = [float(m) for m in matches if float(m) < 100]  # 100m以下のもの
                if lengths:
                    features['length_m'] = max(lengths)
                    features['has_measurement'] = 1
            except ValueError:
                pass
        
        return features
    
    def create_text_features(self, texts: List[str], max_features: int = 1000) -> np.ndarray:
        """TF-IDF特徴量の作成"""
        # テキストの前処理とトークン化
        processed_texts = []
        for text in texts:
            tokens = self.tokenize_japanese(text)
            processed_texts.append(' '.join(tokens))
        
        # TF-IDF vectorizer の初期化
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                token_pattern=r'\S+'  # 空白区切り
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(processed_texts)
        
        return tfidf_matrix.toarray()
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ特徴量の作成"""
        categorical_features = df.copy()
        
        # 年月の特徴量
        if 'InspectionYMD' in df.columns:
            categorical_features['inspection_year'] = pd.to_datetime(df['InspectionYMD']).dt.year
            categorical_features['inspection_month'] = pd.to_datetime(df['InspectionYMD']).dt.month
            categorical_features['inspection_quarter'] = pd.to_datetime(df['InspectionYMD']).dt.quarter
            categorical_features['inspection_season'] = pd.to_datetime(df['InspectionYMD']).dt.month.map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'autumn', 10: 'autumn', 11: 'autumn'
            })
        
        # 橋梁名の特徴量（頻度ベース）
        if 'BridgeName' in df.columns:
            bridge_counts = df['BridgeName'].value_counts()
            categorical_features['bridge_frequency'] = df['BridgeName'].map(bridge_counts)
            categorical_features['bridge_is_common'] = (categorical_features['bridge_frequency'] >= 5).astype(int)
        
        return categorical_features
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str], 
                                  fit: bool = True) -> np.ndarray:
        """カテゴリ特徴量のエンコーディング"""
        encoded_features = []
        
        for col in categorical_columns:
            if col not in df.columns:
                continue
                
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                encoded = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # 未知のカテゴリは-1に設定
                    labels = df[col].astype(str)
                    known_labels = set(self.label_encoders[col].classes_)
                    encoded = []
                    for label in labels:
                        if label in known_labels:
                            encoded.append(self.label_encoders[col].transform([label])[0])
                        else:
                            encoded.append(-1)
                    encoded = np.array(encoded)
                else:
                    encoded = np.zeros(len(df))
            
            encoded_features.append(encoded.reshape(-1, 1))
        
        if encoded_features:
            return np.hstack(encoded_features)
        else:
            return np.array([]).reshape(len(df), 0)
    
    def create_all_features(self, df: pd.DataFrame, text_columns: List[str], 
                           categorical_columns: List[str], numerical_columns: List[str],
                           fit: bool = True) -> Tuple[np.ndarray, List[str]]:
        """全ての特徴量を作成"""
        all_features = []
        feature_names = []
        
        # 1. テキスト特徴量
        combined_text = []
        for _, row in df.iterrows():
            texts = []
            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    texts.append(str(row[col]))
            combined_text.append(' '.join(texts))
        
        if combined_text:
            text_features = self.create_text_features(combined_text)
            all_features.append(text_features)
            if self.tfidf_vectorizer:
                feature_names.extend([f'tfidf_{i}' for i in range(text_features.shape[1])])
        
        # 2. キーワード特徴量
        keyword_features_list = []
        for text in combined_text:
            keyword_feat = self.extract_keyword_features(text)
            keyword_features_list.append(keyword_feat)
        
        if keyword_features_list:
            keyword_df = pd.DataFrame(keyword_features_list)
            keyword_features = keyword_df.values
            all_features.append(keyword_features)
            feature_names.extend(keyword_df.columns.tolist())
        
        # 3. 数値特徴量（テキストから抽出）
        numerical_text_features_list = []
        for text in combined_text:
            num_feat = self.extract_numerical_features(text)
            numerical_text_features_list.append(num_feat)
        
        if numerical_text_features_list:
            numerical_text_df = pd.DataFrame(numerical_text_features_list)
            numerical_text_features = numerical_text_df.values
            all_features.append(numerical_text_features)
            feature_names.extend(numerical_text_df.columns.tolist())
        
        # 4. カテゴリ特徴量
        categorical_df = self.create_categorical_features(df)
        categorical_features = self.encode_categorical_features(
            categorical_df, categorical_columns, fit=fit
        )
        if categorical_features.shape[1] > 0:
            all_features.append(categorical_features)
            feature_names.extend([f'cat_{col}' for col in categorical_columns 
                                if col in categorical_df.columns])
        
        # 5. 既存の数値特徴量
        existing_numerical = []
        for col in numerical_columns:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').fillna(0).values
                existing_numerical.append(values.reshape(-1, 1))
                feature_names.append(f'num_{col}')
        
        if existing_numerical:
            existing_numerical_features = np.hstack(existing_numerical)
            all_features.append(existing_numerical_features)
        
        # 全特徴量の結合
        if all_features:
            combined_features = np.hstack(all_features)
            
            # 数値特徴量の正規化
            if fit:
                combined_features = self.scaler.fit_transform(combined_features)
            else:
                combined_features = self.scaler.transform(combined_features)
            
            self.feature_names = feature_names
            return combined_features, feature_names
        else:
            return np.array([]).reshape(len(df), 0), []

def main():
    """特徴量エンジニアリングのテスト実行"""
    # サンプルデータの作成
    sample_data = pd.DataFrame({
        'Diagnosis': [
            '主桁に鉄筋露出が見られる。間詰め床版に遊離石灰が見られる。',
            '舗装にひびわれが見られる。防護柵に腐食、亀裂が見られる。',
            '下部工にひびわれ、遊離石灰、漏水が見られる。'
        ],
        'DamageComment': [
            '鉄筋が露出しており、鉄筋が腐食している。',
            '舗装にひびわれが見られる。（ひびわれ幅5.0mm)',
            '下部工にひびわれ、遊離石灰が見られる。（ひびわれ幅0.1mm）'
        ],
        'BridgeName': ['山田橋', '岡本橋', '山田橋'],
        'InspectionYMD': ['2015/12/15', '2024/2/15', '2015/12/15'],
        'damage_rank_encoded': [3, 2, 2],
        'HealthLevel': ['Ⅱ', 'Ⅱ', 'Ⅱ']
    })
    
    # 特徴量エンジニアリング実行
    feature_engineer = FeatureEngineer()
    
    features, feature_names = feature_engineer.create_all_features(
        sample_data,
        text_columns=['Diagnosis', 'DamageComment'],
        categorical_columns=['BridgeName', 'inspection_season'],
        numerical_columns=['damage_rank_encoded']
    )
    
    print(f"Features shape: {features.shape}")
    print(f"Feature names count: {len(feature_names)}")
    print(f"First 10 feature names: {feature_names[:10]}")

if __name__ == "__main__":
    main()