
"""
Visualization Module untuk Aplikasi Moneyball
==============================================
Modul ini berisi class dan fungsi untuk membuat
visualisasi statistik pemain sepak bola.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from typing import List, Dict, Optional, Tuple

# DEFINISI FITUR

RADAR_FEATURES = {
    'FW': ['Gls', 'Ast', 'xG', 'xAG', 'Sh', 'SoT', 'PrgC', 'PrgR'],
    'MF': ['Gls', 'Ast', 'xG', 'xAG', 'PrgP', 'Tkl', 'Int', 'PrgC'],
    'DF': ['Tkl', 'TklW', 'Int', 'Clr', 'PrgP', 'PrgC', 'Ast', 'Touches'],
    'ALL': ['Gls', 'Ast', 'xG', 'Tkl', 'Int', 'PrgC', 'PrgP', 'Touches']
}

FEATURE_LABELS = {
    'Gls': 'Goals', 'Ast': 'Assists', 'xG': 'Expected Goals',
    'xAG': 'Expected Assists', 'Sh': 'Shots', 'SoT': 'Shots on Target',
    'PrgC': 'Progressive Carries', 'PrgP': 'Progressive Passes',
    'PrgR': 'Progressive Receives', 'Tkl': 'Tackles', 'TklW': 'Tackles Won',
    'Int': 'Interceptions', 'Clr': 'Clearances', 'Touches': 'Touches'
}


class PlayerVisualizer:
    """
    Class untuk membuat visualisasi statistik pemain sepak bola.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._add_main_position()
        
    def _add_main_position(self):
        if 'Pos_Main' not in self.df.columns:
            self.df['Pos_Main'] = self.df['Pos'].apply(
                lambda x: x.split(',')[0].strip() if isinstance(x, str) else 'Unknown'
            )
    
    def find_player(self, player_name: str) -> Optional[pd.Series]:
        matches = self.df[self.df['Player'].str.contains(player_name, case=False, na=False)]
        if len(matches) == 0:
            return None
        return matches.iloc[0]
    
    def get_percentile_rank(self, player_name: str, features: List[str], position: str = None) -> Dict:
        player = self.find_player(player_name)
        if player is None:
            return {}
        
        if position:
            compare_df = self.df[self.df['Pos_Main'] == position.upper()]
        else:
            compare_df = self.df
        
        percentiles = {}
        for feat in features:
            if feat in self.df.columns:
                player_val = player[feat]
                percentile = (compare_df[feat] <= player_val).mean() * 100
                percentiles[feat] = percentile
        
        return percentiles
    
    def radar_chart(self, player_name: str, features: List[str] = None,
                    use_percentile: bool = True, position: str = None,
                    figsize: Tuple = (8, 8), color: str = '#1f77b4') -> plt.Figure:
        player = self.find_player(player_name)
        if player is None:
            return None
        
        if features is None:
            pos = position or player['Pos_Main']
            features = RADAR_FEATURES.get(pos, RADAR_FEATURES['ALL'])
        features = [f for f in features if f in self.df.columns]
        
        if use_percentile:
            percentiles = self.get_percentile_rank(player_name, features, position)
            values = [percentiles.get(f, 0) for f in features]
        else:
            values = [player[f] for f in features]
        
        labels = [FEATURE_LABELS.get(f, f) for f in features]
        
        num_vars = len(features)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=10)
        
        if use_percentile:
            ax.set_ylim(0, 100)
        
        title = f"{player['Player']} - {player['Squad']} ({player['Pos']})"
        ax.set_title(title, size=14, fontweight='bold', y=1.08)
        plt.tight_layout()
        return fig
    
    def radar_chart_comparison(self, player1: str, player2: str,
                               features: List[str] = None, use_percentile: bool = True,
                               figsize: Tuple = (10, 10),
                               colors: List[str] = ['#1f77b4', '#ff7f0e']) -> plt.Figure:
        p1 = self.find_player(player1)
        p2 = self.find_player(player2)
        if p1 is None or p2 is None:
            return None
        
        if features is None:
            features = RADAR_FEATURES['ALL']
        features = [f for f in features if f in self.df.columns]
        
        if use_percentile:
            pct1 = self.get_percentile_rank(player1, features)
            pct2 = self.get_percentile_rank(player2, features)
            values1 = [pct1.get(f, 0) for f in features]
            values2 = [pct2.get(f, 0) for f in features]
        else:
            values1 = [p1[f] for f in features]
            values2 = [p2[f] for f in features]
        
        labels = [FEATURE_LABELS.get(f, f) for f in features]
        
        num_vars = len(features)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        values1 += values1[:1]
        values2 += values2[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        ax.plot(angles, values1, 'o-', linewidth=2, color=colors[0], label=p1['Player'])
        ax.fill(angles, values1, alpha=0.2, color=colors[0])
        ax.plot(angles, values2, 'o-', linewidth=2, color=colors[1], label=p2['Player'])
        ax.fill(angles, values2, alpha=0.2, color=colors[1])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=10)
        if use_percentile:
            ax.set_ylim(0, 100)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        title = f"{p1['Player']} vs {p2['Player']}"
        ax.set_title(title, size=14, fontweight='bold', y=1.08)
        plt.tight_layout()
        return fig
    
    def bar_chart_stats(self, player_name: str, features: List[str] = None,
                        figsize: Tuple = (12, 6), color: str = '#1f77b4') -> plt.Figure:
        player = self.find_player(player_name)
        if player is None:
            return None
        
        if features is None:
            features = ['Gls', 'Ast', 'G+A', 'xG', 'xAG', 'Sh', 'SoT', 'PrgC', 'PrgP']
        features = [f for f in features if f in self.df.columns]
        
        values = [player[f] for f in features]
        labels = [FEATURE_LABELS.get(f, f) for f in features]
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(labels, values, color=color, edgecolor='black', alpha=0.8)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f"{player['Player']} - {player['Squad']}", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def bar_chart_comparison(self, player1: str, player2: str,
                             features: List[str] = None, figsize: Tuple = (14, 7),
                             colors: List[str] = ['#1f77b4', '#ff7f0e']) -> plt.Figure:
        p1 = self.find_player(player1)
        p2 = self.find_player(player2)
        if p1 is None or p2 is None:
            return None
        
        if features is None:
            features = ['Gls', 'Ast', 'G+A', 'xG', 'xAG', 'Sh', 'SoT', 'PrgC']
        features = [f for f in features if f in self.df.columns]
        
        values1 = [p1[f] for f in features]
        values2 = [p2[f] for f in features]
        labels = [FEATURE_LABELS.get(f, f) for f in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width/2, values1, width, label=p1['Player'], color=colors[0], alpha=0.8)
        ax.bar(x + width/2, values2, width, label=p2['Player'], color=colors[1], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.set_title(f"{p1['Player']} vs {p2['Player']}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def pie_chart_distribution(self, column: str, figsize: Tuple = (10, 8),
                               title: str = None, top_n: int = None) -> plt.Figure:
        counts = self.df[column].value_counts()
        if top_n:
            counts = counts.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.axis('equal')
        ax.set_title(title or f"Distribusi {column}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def top_players_chart(self, feature: str, top_n: int = 10, position: str = None,
                          figsize: Tuple = (12, 8), color: str = '#2ecc71') -> plt.Figure:
        df = self.df.copy()
        if position:
            df = df[df['Pos_Main'] == position.upper()]
        
        top_players = df.nlargest(top_n, feature)[['Player', 'Squad', feature]]
        
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(top_players))
        bars = ax.barh(y_pos, top_players[feature].values, color=color, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['Player']} ({row['Squad']})" 
                           for _, row in top_players.iterrows()])
        ax.invert_yaxis()
        
        label = FEATURE_LABELS.get(feature, feature)
        ax.set_xlabel(label, fontsize=12)
        title = f"Top {top_n} Pemain - {label}"
        if position:
            title += f" ({position})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
