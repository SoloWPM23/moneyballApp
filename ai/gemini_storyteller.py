"""
Gemini AI Storyteller Module untuk Aplikasi Moneyball
======================================================
Modul ini menggunakan Google Gemini AI (google-genai) untuk menghasilkan
narasi dan analisis tentang pemain sepak bola.
"""

import json
import time
import pandas as pd
from typing import Dict, List, Optional

try:
    from google import genai
except ImportError:
    raise ImportError("Please install google-genai: pip install google-genai")


class GeminiStoryteller:
    """
    Class untuk menghasilkan narasi AI tentang pemain sepak bola
    menggunakan Google Gemini API (google-genai).
    """
    
    def __init__(self, api_key: str, df: pd.DataFrame, model_name: str = 'gemini-2.5-flash'):
        """
        Inisialisasi GeminiStoryteller.
        
        Args:
            api_key: Google Gemini API key
            df: DataFrame dengan data pemain
            model_name: Nama model Gemini (default: gemini-2.5-flash untuk free tier)
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.df = df.copy()
        self._add_main_position()
        
        self.system_context = """
        Kamu adalah seorang analis sepak bola profesional dan scout berpengalaman.
        Tugas kamu adalah memberikan analisis mendalam tentang pemain sepak bola
        berdasarkan statistik yang diberikan. Gunakan bahasa Indonesia yang baik
        dan mudah dipahami. Berikan insight yang berguna untuk manajer dan scout.
        """
    
    def _add_main_position(self):
        """Tambah kolom Pos_Main jika belum ada."""
        if 'Pos_Main' not in self.df.columns:
            self.df['Pos_Main'] = self.df['Pos'].apply(
                lambda x: x.split(',')[0].strip() if isinstance(x, str) else 'Unknown'
            )
    
    def find_player(self, player_name: str) -> Optional[pd.Series]:
        """Cari data pemain berdasarkan nama."""
        matches = self.df[self.df['Player'].str.contains(player_name, case=False, na=False)]
        if len(matches) == 0:
            return None
        return matches.iloc[0]
    
    def _get_player_stats_text(self, player: pd.Series) -> str:
        """Format statistik pemain menjadi text."""
        stats = f"""
        Nama: {player['Player']}
        Klub: {player['Squad']}
        Liga: {player['Comp']}
        Posisi: {player['Pos']}
        
        Statistik Utama:
        - Goals: {player.get('Gls', 0)}
        - Assists: {player.get('Ast', 0)}
        - Goal + Assist: {player.get('G+A', 0)}
        - Expected Goals (xG): {player.get('xG', 0):.2f}
        - Expected Assists (xAG): {player.get('xAG', 0):.2f}
        - Shots: {player.get('Sh', 0)}
        - Shots on Target: {player.get('SoT', 0)}
        - Progressive Carries: {player.get('PrgC', 0)}
        - Progressive Passes: {player.get('PrgP', 0)}
        - Tackles: {player.get('Tkl', 0)}
        - Interceptions: {player.get('Int', 0)}
        - Minutes Played: {player.get('Min', 0)}
        """
        return stats
    
    def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """
        Call Gemini API dengan retry logic untuk handle rate limits.
        
        Args:
            prompt: Prompt untuk Gemini
            max_retries: Jumlah maksimum retry jika rate limited
            
        Returns:
            Response text dari Gemini
        """
        full_prompt = f"{self.system_context}\n\n{prompt}"
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = (attempt + 1) * 15  # 15s, 30s, 45s
                    print(f"⏳ Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    return f"Error generating content: {error_str}"
        
        return "Rate limit exceeded. Please wait a few minutes and try again."
    
    def generate_player_description(self, player_name: str) -> str:
        """Generate deskripsi lengkap tentang seorang pemain."""
        player = self.find_player(player_name)
        if player is None:
            return f"Pemain '{player_name}' tidak ditemukan dalam database."
        
        stats = self._get_player_stats_text(player)
        
        prompt = f"""
        Berdasarkan statistik berikut, buatlah profil pemain yang menarik dan informatif:
        
        {stats}
        
        Format output:
        1. **Overview**: Deskripsi singkat tentang pemain dan gaya bermainnya
        2. **Kekuatan**: 3 poin kekuatan utama berdasarkan statistik
        3. **Area Pengembangan**: 2 area yang bisa ditingkatkan
        4. **Kesimpulan**: Satu paragraf tentang potensi dan value pemain
        
        Gunakan bahasa Indonesia yang engaging dan mudah dipahami.
        """
        
        return self._call_gemini(prompt)
    
    def generate_comparison_narrative(self, player1_name: str, player2_name: str) -> str:
        """Generate narasi perbandingan antara dua pemain."""
        player1 = self.find_player(player1_name)
        player2 = self.find_player(player2_name)
        
        if player1 is None:
            return f"Pemain '{player1_name}' tidak ditemukan."
        if player2 is None:
            return f"Pemain '{player2_name}' tidak ditemukan."
        
        stats1 = self._get_player_stats_text(player1)
        stats2 = self._get_player_stats_text(player2)
        
        prompt = f"""
        Bandingkan kedua pemain berikut secara mendalam:
        
        === PEMAIN 1 ===
        {stats1}
        
        === PEMAIN 2 ===
        {stats2}
        
        Format analisis:
        1. **Head-to-Head**: Tabel perbandingan statistik kunci
        2. **Analisis Komparatif**: Perbandingan gaya bermain dan kontribusi
        3. **Keunggulan Masing-masing**: Dimana tiap pemain lebih unggul
        4. **Verdict**: Kesimpulan siapa yang lebih cocok untuk situasi tertentu
        
        Berikan analisis objektif berdasarkan data, bukan opini subjektif.
        """
        
        return self._call_gemini(prompt)
    
    def generate_recommendation_explanation(
        self, 
        target_player: str, 
        similar_players: List[Dict],
        search_criteria: Dict = None
    ) -> str:
        """Generate penjelasan mengapa pemain-pemain tertentu direkomendasikan."""
        target = self.find_player(target_player)
        if target is None:
            return f"Pemain '{target_player}' tidak ditemukan."
        
        target_stats = self._get_player_stats_text(target)
        
        # Build similar players info
        similar_info = []
        for sp in similar_players[:5]:
            player = self.find_player(sp['name'])
            if player is not None:
                similar_info.append({
                    'name': sp['name'],
                    'score': sp['similarity_score'],
                    'club': player['Squad'],
                    'position': player['Pos'],
                    'goals': player.get('Gls', 0),
                    'assists': player.get('Ast', 0)
                })
        
        similar_text = "\n".join([
            f"- {p['name']} ({p['club']}) - Similarity: {p['score']:.1%}, G: {p['goals']}, A: {p['assists']}"
            for p in similar_info
        ])
        
        criteria_text = ""
        if search_criteria:
            criteria_text = f"\nKriteria pencarian: {search_criteria}"
        
        prompt = f"""
        Jelaskan hasil rekomendasi pemain berikut untuk tim yang mencari pengganti/alternatif:
        
        === PEMAIN TARGET ===
        {target_stats}
        
        === PEMAIN YANG DIREKOMENDASIKAN ===
        {similar_text}
        {criteria_text}
        
        Format penjelasan:
        1. **Mengapa Rekomendasi Ini**: Jelaskan logika di balik rekomendasi
        2. **Profil Singkat**: Deskripsi singkat tiap pemain yang direkomendasikan
        3. **Fit Analysis**: Seberapa cocok mereka sebagai pengganti/alternatif
        4. **Saran Scout**: Rekomendasi prioritas untuk di-scout lebih lanjut
        
        Gunakan bahasa Indonesia yang profesional namun mudah dipahami.
        """
        
        return self._call_gemini(prompt)
    
    def generate_scout_report(self, player_name: str) -> str:
        """Generate laporan scout profesional untuk seorang pemain."""
        player = self.find_player(player_name)
        if player is None:
            return f"Pemain '{player_name}' tidak ditemukan."
        
        stats = self._get_player_stats_text(player)
        
        prompt = f"""
        Buatlah laporan scout profesional untuk pemain berikut:
        
        {stats}
        
        Format laporan scout:
        
        **LAPORAN SCOUT**
        
        **1. INFORMASI DASAR**
        - Ringkasan profil pemain
        
        **2. ANALISIS TEKNIS**
        - Kemampuan mencetak gol
        - Kemampuan kreativitas/assist
        - Kontribusi defensif
        - Kemampuan progresif (membawa dan mengoper bola maju)
        
        **3. RATING ASPEK** (skala 1-10)
        - Finishing: X/10
        - Playmaking: X/10
        - Defensive Work: X/10
        - Physical Presence: X/10
        
        **4. REKOMENDASI**
        - Tipe tim yang cocok
        - Estimasi nilai transfer (berdasarkan performa)
        - Potensi pengembangan
        
        **5. KESIMPULAN AKHIR**
        - Summary dalam 2-3 kalimat
        
        Jadikan laporan ini seperti laporan scout profesional sesungguhnya.
        """
        
        return self._call_gemini(prompt)
    
    def quick_summary(self, player_name: str) -> str:
        """Generate ringkasan singkat tentang pemain (untuk tooltip/preview)."""
        player = self.find_player(player_name)
        if player is None:
            return "Pemain tidak ditemukan."
        
        stats = self._get_player_stats_text(player)
        
        prompt = f"""
        Berikan ringkasan SANGAT SINGKAT (maksimal 3 kalimat) tentang pemain ini:
        
        {stats}
        
        Fokus pada: posisi, gaya bermain utama, dan satu highlight statistik.
        Jawab dalam format paragraf singkat saja, tanpa bullet point.
        """
        
        return self._call_gemini(prompt)


def load_api_key(config_path: str = 'data/config.json') -> str:
    """Load API key dari config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get('API_KEY', '')


def load_api_key(config_path: str) -> str:
    """Load API key from config file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('API_KEY', '')
    except Exception:
        return ''


def load_storyteller(config_path: str, data_path: str) -> GeminiStoryteller:
    """Helper function to load storyteller with config."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    df = pd.read_csv(data_path)
    api_key = config.get('API_KEY')
    
    return GeminiStoryteller(api_key, df)


# For testing
if __name__ == "__main__":
    print("GeminiStoryteller module loaded successfully!")
    print("Usage:")
    print("  from ai.gemini_storyteller import GeminiStoryteller, load_api_key")
    print("  api_key = load_api_key('data/config.json')")
    print("  storyteller = GeminiStoryteller(api_key, df)")
