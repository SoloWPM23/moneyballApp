
"""
Similarity Model Module untuk Aplikasi Moneyball
=================================================
Modul ini berisi class dan fungsi untuk menghitung
kemiripan pemain sepak bola berdasarkan statistik performa.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple, Optional

# DEFINISI FITUR BERDASARKAN POSISI

FW_FEATURES = [
    'Gls', 'Ast', 'G+A', 'xG', 'xAG', 'npxG', 'G-PK',
    'Sh', 'SoT', 'Sh/90', 'SoT/90', 'G-xG',
    'Touches', 'Carries', 'PrgC', 'PrgR',
    'KP', 'PPA', 'CrsPA', '90s'
]

MF_FEATURES = [
    'Gls', 'Ast', 'G+A', 'xG', 'xAG',
    'Cmp', 'Att', 'TotDist', 'PrgDist', 'KP', '1/3', 'PPA', 'PrgP',
    'Tkl', 'TklW', 'Int',
    'Touches', 'Carries', 'PrgC', 'PrgR', '90s'
]

DF_FEATURES = [
    'Tkl', 'TklW', 'Def 3rd', 'Mid 3rd', 'Int', 'Clr', 'Err',
    'Cmp', 'Att', 'TotDist', 'PrgDist', 'PrgP',
    'Touches', 'Carries', 'PrgC', 'Ast', 'xAG', '90s'
]

COMMON_FEATURES = [
    'Gls', 'Ast', 'G+A', 'xG', 'xAG',
    'Cmp', 'Att', 'PrgP',
    'Tkl', 'TklW', 'Int',
    'Touches', 'Carries', 'PrgC', 'PrgR', '90s'
]


class PlayerSimilarityModel:
    """
    Model untuk menghitung kemiripan pemain sepak bola.
    """
    
    def __init__(self, df: pd.DataFrame, scaler_type: str = 'standard'):
        self.df = df.copy()
        self.scaler_type = scaler_type
        self.scaler = None
        self.features = None
        self.normalized_data = None
        self.knn_model = None
        self.similarity_matrix = None
        self._add_main_position()
        
    def _add_main_position(self):
        self.df['Pos_Main'] = self.df['Pos'].apply(
            lambda x: x.split(',')[0].strip() if isinstance(x, str) else 'Unknown'
        )
    
    def get_features_for_position(self, position: str) -> List[str]:
        position = position.upper()
        if position == 'FW':
            return FW_FEATURES
        elif position == 'MF':
            return MF_FEATURES
        elif position == 'DF':
            return DF_FEATURES
        else:
            return COMMON_FEATURES
    
    def filter_available_features(self, features: List[str]) -> List[str]:
        return [f for f in features if f in self.df.columns]
    
    def prepare_data(
        self, 
        features: Optional[List[str]] = None,
        position: Optional[str] = None,
        min_minutes: int = 0
    ) -> np.ndarray:
        df = self.df.copy()
        
        if min_minutes > 0 and 'Min' in df.columns:
            df = df[df['Min'] >= min_minutes]
        
        if position:
            df = df[df['Pos_Main'] == position.upper()]
        
        self.df = df.reset_index(drop=True)
        
        if features is None:
            if position:
                features = self.get_features_for_position(position)
            else:
                features = COMMON_FEATURES
        
        self.features = self.filter_available_features(features)
        self.df[self.features] = self.df[self.features].fillna(0)
        
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.normalized_data = self.scaler.fit_transform(self.df[self.features])
        return self.normalized_data
    
    def compute_similarity_matrix(self, method: str = 'cosine') -> np.ndarray:
        if self.normalized_data is None:
            raise ValueError("Data belum dipersiapkan.")
        
        if method == 'cosine':
            self.similarity_matrix = cosine_similarity(self.normalized_data)
        else:
            distances = euclidean_distances(self.normalized_data)
            self.similarity_matrix = 1 / (1 + distances)
        
        return self.similarity_matrix
    
    def fit_knn(self, n_neighbors: int = 11, metric: str = 'cosine'):
        if self.normalized_data is None:
            raise ValueError("Data belum dipersiapkan.")
        
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors, 
            metric=metric,
            algorithm='brute'
        )
        self.knn_model.fit(self.normalized_data)
    
    def find_player_index(self, player_name: str) -> Optional[int]:
        matches = self.df[self.df['Player'].str.contains(player_name, case=False, na=False)]
        if len(matches) == 0:
            return None
        return matches.index[0]
    
    def get_similar_players(
        self, 
        player_name: str, 
        top_n: int = 10,
        method: str = 'cosine'
    ) -> pd.DataFrame:
        if method == 'knn':
            return self._get_similar_knn(player_name, top_n)
        return self._get_similar_cosine(player_name, top_n)
    
    def _get_similar_cosine(self, player_name: str, top_n: int) -> pd.DataFrame:
        if self.similarity_matrix is None:
            self.compute_similarity_matrix(method='cosine')
        
        player_idx = self.find_player_index(player_name)
        if player_idx is None:
            return pd.DataFrame()
        
        similarities = self.similarity_matrix[player_idx]
        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        
        results = self.df.iloc[similar_indices][['Player', 'Pos', 'Squad', 'Comp']].copy()
        results['Similarity'] = similarities[similar_indices]
        results['Rank'] = range(1, len(results) + 1)
        return results[['Rank', 'Player', 'Pos', 'Squad', 'Comp', 'Similarity']]
    
    def _get_similar_knn(self, player_name: str, top_n: int) -> pd.DataFrame:
        if self.knn_model is None:
            self.fit_knn(n_neighbors=top_n + 1)
        
        player_idx = self.find_player_index(player_name)
        if player_idx is None:
            return pd.DataFrame()
        
        player_vector = self.normalized_data[player_idx].reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(player_vector)
        
        similar_indices = indices[0][1:top_n+1]
        similar_distances = distances[0][1:top_n+1]
        
        results = self.df.iloc[similar_indices][['Player', 'Pos', 'Squad', 'Comp']].copy()
        results['Similarity'] = 1 - similar_distances
        results['Rank'] = range(1, len(results) + 1)
        return results[['Rank', 'Player', 'Pos', 'Squad', 'Comp', 'Similarity']]
    
    def get_player_stats(self, player_name: str) -> Optional[pd.Series]:
        player_idx = self.find_player_index(player_name)
        if player_idx is None:
            return None
        return self.df.iloc[player_idx]
    
    def search_players(self, query: str) -> pd.DataFrame:
        return self.df[self.df['Player'].str.contains(query, case=False, na=False)][
            ['Player', 'Pos', 'Squad', 'Comp', 'Gls', 'Ast', 'Min']
        ]
    
    def get_player_comparison(self, player1: str, player2: str) -> pd.DataFrame:
        stats1 = self.get_player_stats(player1)
        stats2 = self.get_player_stats(player2)
        
        if stats1 is None or stats2 is None:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'Stat': self.features,
            player1: [stats1[f] for f in self.features],
            player2: [stats2[f] for f in self.features]
        })
