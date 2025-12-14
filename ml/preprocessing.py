"""
Data Preprocessing Module untuk Aplikasi Moneyball
===================================================
Modul ini berisi fungsi-fungsi untuk preprocessing data pemain sepak bola
sebelum digunakan dalam model similarity.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Dict, Tuple, Optional


# DEFINISI FITUR BERDASARKAN KATEGORI

# Fitur untuk Forward (FW)
FW_FEATURES = [
    'Gls', 'Ast', 'G+A', 'xG', 'xAG', 'npxG', 'G-PK',
    'Sh', 'SoT', 'Sh/90', 'SoT/90', 'G-xG',
    'Touches', 'Carries', 'PrgC', 'PrgR',
    'KP', 'PPA', 'CrsPA',
    '90s'
]

# Fitur untuk Midfielder (MF)
MF_FEATURES = [
    'Gls', 'Ast', 'G+A', 'xG', 'xAG',
    'Cmp', 'Att', 'TotDist', 'PrgDist', 'KP', '1/3', 'PPA', 'PrgP',
    'Tkl', 'TklW', 'Int',
    'Touches', 'Carries', 'PrgC', 'PrgR',
    '90s'
]

# Fitur untuk Defender (DF)
DF_FEATURES = [
    'Tkl', 'TklW', 'Def 3rd', 'Mid 3rd', 'Int', 'Clr', 'Err',
    'Cmp', 'Att', 'TotDist', 'PrgDist', 'PrgP',
    'Touches', 'Carries', 'PrgC',
    'Ast', 'xAG',
    '90s'
]

# Fitur untuk Goalkeeper (GK) - jika tersedia
GK_FEATURES = [
    '90s', 'MP', 'Starts', 'Min'
]

# Fitur umum untuk semua posisi
COMMON_FEATURES = [
    'Gls', 'Ast', 'G+A', 'xG', 'xAG',
    'Cmp', 'Att', 'PrgP',
    'Tkl', 'TklW', 'Int',
    'Touches', 'Carries', 'PrgC', 'PrgR',
    '90s'
]


class DataPreprocessor:
    """
    Class untuk preprocessing data pemain sepak bola.
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Inisialisasi preprocessor.
        
        Args:
            scaler_type: 'standard' untuk StandardScaler, 'minmax' untuk MinMaxScaler
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_columns = None
        self.df_original = None
        self.df_processed = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data dari file CSV.
        
        Args:
            filepath: Path ke file CSV
            
        Returns:
            DataFrame dengan data pemain
        """
        self.df_original = pd.read_csv(filepath)
        return self.df_original.copy()
    
    def get_main_position(self, pos: str) -> str:
        """
        Ekstrak posisi utama dari string posisi.
        
        Args:
            pos: String posisi (bisa multiple, e.g., "FW,MF")
            
        Returns:
            Posisi utama
        """
        if pd.isna(pos):
            return 'Unknown'
        return pos.split(',')[0].strip()
    
    def add_main_position_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tambahkan kolom posisi utama ke dataframe.
        
        Args:
            df: DataFrame dengan kolom 'Pos'
            
        Returns:
            DataFrame dengan kolom 'Pos_Main' baru
        """
        df = df.copy()
        df['Pos_Main'] = df['Pos'].apply(self.get_main_position)
        return df
    
    def get_features_for_position(self, position: str) -> List[str]:
        """
        Dapatkan list fitur berdasarkan posisi pemain.
        
        Args:
            position: Posisi pemain (FW, MF, DF, GK)
            
        Returns:
            List nama fitur
        """
        position = position.upper()
        
        if position == 'FW':
            return FW_FEATURES
        elif position == 'MF':
            return MF_FEATURES
        elif position == 'DF':
            return DF_FEATURES
        elif position == 'GK':
            return GK_FEATURES
        else:
            return COMMON_FEATURES
    
    def filter_available_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        Filter fitur yang tersedia di dataframe.
        
        Args:
            df: DataFrame
            features: List fitur yang diinginkan
            
        Returns:
            List fitur yang tersedia
        """
        return [f for f in features if f in df.columns]
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values dalam dataframe.
        
        Args:
            df: DataFrame
            strategy: 'mean', 'median', 'zero', atau 'drop'
            
        Returns:
            DataFrame tanpa missing values
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        elif strategy == 'drop':
            df = df.dropna(subset=numeric_cols)
            
        return df
    
    def normalize_features(self, df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Normalisasi fitur numerik.
        
        Args:
            df: DataFrame
            features: List fitur untuk dinormalisasi
            
        Returns:
            Tuple (DataFrame dengan fitur ternormalisasi, array data ternormalisasi)
        """
        df = df.copy()
        available_features = self.filter_available_features(df, features)
        
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        # Fit dan transform
        normalized_data = self.scaler.fit_transform(df[available_features])
        
        # Buat dataframe baru dengan data ternormalisasi
        normalized_df = pd.DataFrame(
            normalized_data,
            columns=[f"{col}_norm" for col in available_features],
            index=df.index
        )
        
        # Gabungkan dengan kolom non-numerik
        result_df = pd.concat([df, normalized_df], axis=1)
        
        self.feature_columns = available_features
        
        return result_df, normalized_data
    
    def prepare_for_similarity(
        self, 
        df: pd.DataFrame, 
        position: Optional[str] = None,
        features: Optional[List[str]] = None,
        min_minutes: int = 0
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Persiapkan data untuk perhitungan similarity.
        
        Args:
            df: DataFrame asli
            position: Filter berdasarkan posisi (optional)
            features: Custom features (optional)
            min_minutes: Minimum menit bermain
            
        Returns:
            Tuple (DataFrame terfilter, array fitur ternormalisasi)
        """
        df = df.copy()
        
        # Tambah kolom posisi utama
        df = self.add_main_position_column(df)
        
        # Filter berdasarkan menit minimum
        if min_minutes > 0 and 'Min' in df.columns:
            df = df[df['Min'] >= min_minutes]
        
        # Filter berdasarkan posisi jika diperlukan
        if position:
            df = df[df['Pos_Main'] == position.upper()]
        
        # Tentukan fitur yang akan digunakan
        if features is None:
            if position:
                features = self.get_features_for_position(position)
            else:
                features = COMMON_FEATURES
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy='mean')
        
        # Normalisasi
        df, normalized_data = self.normalize_features(df, features)
        
        self.df_processed = df
        
        return df, normalized_data
    
    def get_player_data(self, df: pd.DataFrame, player_name: str) -> Optional[pd.Series]:
        """
        Dapatkan data pemain berdasarkan nama.
        
        Args:
            df: DataFrame
            player_name: Nama pemain
            
        Returns:
            Series data pemain atau None jika tidak ditemukan
        """
        matches = df[df['Player'].str.contains(player_name, case=False, na=False)]
        
        if len(matches) == 0:
            return None
        elif len(matches) == 1:
            return matches.iloc[0]
        else:
            # Jika ada multiple match, kembalikan yang pertama
            return matches.iloc[0]
    
    def search_players(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Cari pemain berdasarkan nama.
        
        Args:
            df: DataFrame
            query: Query pencarian
            
        Returns:
            DataFrame hasil pencarian
        """
        return df[df['Player'].str.contains(query, case=False, na=False)]
    
    def filter_by_criteria(
        self,
        df: pd.DataFrame,
        position: Optional[str] = None,
        squad: Optional[str] = None,
        competition: Optional[str] = None,
        min_minutes: int = 0
    ) -> pd.DataFrame:
        """
        Filter dataframe berdasarkan kriteria.
        
        Args:
            df: DataFrame
            position: Filter posisi
            squad: Filter klub
            competition: Filter liga
            min_minutes: Minimum menit bermain
            
        Returns:
            DataFrame terfilter
        """
        df = df.copy()
        
        if 'Pos_Main' not in df.columns:
            df = self.add_main_position_column(df)
        
        if position:
            df = df[df['Pos_Main'] == position.upper()]
        
        if squad:
            df = df[df['Squad'].str.contains(squad, case=False, na=False)]
        
        if competition:
            df = df[df['Comp'].str.contains(competition, case=False, na=False)]
        
        if min_minutes > 0 and 'Min' in df.columns:
            df = df[df['Min'] >= min_minutes]
        
        return df
    
    def get_unique_values(self, df: pd.DataFrame, column: str) -> List[str]:
        """
        Dapatkan nilai unik dari kolom.
        
        Args:
            df: DataFrame
            column: Nama kolom
            
        Returns:
            List nilai unik
        """
        if column in df.columns:
            return sorted(df[column].dropna().unique().tolist())
        return []


# FUNGSI HELPER

def load_and_preprocess(
    filepath: str,
    position: Optional[str] = None,
    min_minutes: int = 0
) -> Tuple[pd.DataFrame, np.ndarray, DataPreprocessor]:
    """
    Fungsi helper untuk load dan preprocess data dalam satu langkah.
    
    Args:
        filepath: Path ke file CSV
        position: Filter posisi (optional)
        min_minutes: Minimum menit bermain
        
    Returns:
        Tuple (DataFrame, array ternormalisasi, preprocessor instance)
    """
    preprocessor = DataPreprocessor(scaler_type='standard')
    df = preprocessor.load_data(filepath)
    df_processed, normalized_data = preprocessor.prepare_for_similarity(
        df, position=position, min_minutes=min_minutes
    )
    
    return df_processed, normalized_data, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    import os
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataPlayerC.csv')
    
    if os.path.exists(data_path):
        print("Testing DataPreprocessor...")
        
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(data_path)
        
        print(f"\nDataset loaded: {len(df)} pemain")
        print(f"Kolom: {len(df.columns)}")
        
        # Test prepare for similarity
        df_processed, normalized = preprocessor.prepare_for_similarity(df, min_minutes=500)
        
        print(f"\nSetelah filter (min 500 menit): {len(df_processed)} pemain")
        print(f"Shape normalized data: {normalized.shape}")
        
        # Test search
        results = preprocessor.search_players(df, "Messi")
        print(f"\nHasil pencarian 'Messi': {len(results)} pemain")
        
        print("\nPreprocessing test completed!")
    else:
        print(f"File tidak ditemukan: {data_path}")
