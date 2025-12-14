# ⚽ Moneyball - Football Player Analysis

Professional football player analysis and scouting tool with AI-powered insights.

## Features

- 🔍 **Player Search** - Search and compare players across 5 major leagues
- 💡 **Similarity Finder** - Find similar players using ML algorithms
- 📊 **Visualizations** - Radar charts, bar charts, scatter plots
- 🤖 **AI Insights** - Generate player profiles and comparisons using Gemini AI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/moneyball-app.git
cd moneyball-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup API Key (for AI features):
   - Create `data/config.json` file:
```json
{
    "API_KEY": "your-gemini-api-key-here"
}
```
   - Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)

4. Run the application:
```bash
python run_app.py
```

## Dataset

Contains 2,854 players from 5 top leagues:
- Premier League
- La Liga
- Bundesliga
- Serie A
- Ligue 1

## Tech Stack

- **GUI**: PyQt6
- **ML**: scikit-learn (cosine similarity, KNN)
- **Visualization**: Matplotlib
- **AI**: Google Gemini AI (gemini-2.5-flash)
- **Data**: pandas, numpy

## Screenshots
<img width="1919" height="990" alt="Screenshot 2025-12-14 224918" src="https://github.com/user-attachments/assets/d1ebb1bf-1e38-44ec-9119-eb5c80b57b1e" />
<img width="1919" height="990" alt="Screenshot 2025-12-14 224932" src="https://github.com/user-attachments/assets/f4e30bc2-6109-45c3-a0ce-df0878bc71e0" />
<img width="1919" height="990" alt="Screenshot 2025-12-14 224942" src="https://github.com/user-attachments/assets/3c7afd3a-5c80-4413-bb36-0c111636e0e2" />
<img width="1919" height="990" alt="Screenshot 2025-12-14 224956" src="https://github.com/user-attachments/assets/465850ed-bcfe-4c73-b74a-3b20b0c690ef" />





