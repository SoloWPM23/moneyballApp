<div align="center">

# ⚽ Moneyball

### Football Player Analysis & Scouting Tool

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://pypi.org/project/PyQt6/)
[![Gemini AI](https://img.shields.io/badge/Gemini_AI-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)

*A professional-grade football analytics application powered by Machine Learning and Google Gemini AI for intelligent player scouting and comparison.*

[Features](#-features) •
[Installation](#-installation) •
[Screenshots](#-screenshots) •
[Tech Stack](#-tech-stack)

</div>

---

## ✨ Features

### 🔍 Player Search
- Search players by name from 2,854 players across 5 major leagues
- View detailed player statistics (goals, assists, passes, tackles, etc.)
- Compare two players side-by-side with statistical breakdown

### 💡 Player Recommendations
- Find similar players using Machine Learning (Cosine Similarity)
- Get top 10 most similar players based on playing style
- Discover alternative players for scouting and recruitment

### 📊 Data Visualization
- **Radar Chart** - Compare player attributes visually
- **Bar Chart** - Side-by-side statistical comparison
- **Scatter Plot** - Analyze goals vs assists distribution
- **Box Plot** - View statistical distribution across positions

### 🤖 AI-Powered Analysis
- Generate professional player profile descriptions
- Create detailed comparison narratives between two players
- Produce scout reports with strengths, weaknesses, and recommendations
- Get quick summary insights for any player
- Powered by Google Gemini 2.5 Flash AI

### 📋 Player Statistics Available
- **Basic Info**: Name, Age, Nationality, Club, League, Position
- **Offensive**: Goals, Assists, Shots, Shot Accuracy, Expected Goals (xG)
- **Passing**: Pass Completion, Key Passes, Progressive Passes
- **Defensive**: Tackles, Interceptions, Clearances, Blocks
- **Physical**: Minutes Played, Matches, Distance Covered

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/SoloWPM23/moneyballApp.git
cd moneyballApp

# Install dependencies
pip install -r requirements.txt

# Run the application
python run_app.py
```

### 🔑 API Configuration (Optional - for AI features)

1. Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Create `data/config.json`:

```json
{
    "API_KEY": "your-gemini-api-key-here"
}
```

> **Note:** The app works without an API key, but AI features will be disabled.

---

## 📸 Screenshots

<div align="center">

| Player Search | Recommendations |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/d1ebb1bf-1e38-44ec-9119-eb5c80b57b1e" width="400"/> | <img src="https://github.com/user-attachments/assets/f4e30bc2-6109-45c3-a0ce-df0878bc71e0" width="400"/> |

| Analysis Charts | AI Insights |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/3c7afd3a-5c80-4413-bb36-0c111636e0e2" width="400"/> | <img src="https://github.com/user-attachments/assets/465850ed-bcfe-4c73-b74a-3b20b0c690ef" width="400"/> |

</div>

---

## 🗂️ Dataset

The application includes comprehensive data from **5 major European leagues**:

| League | Country | Players |
|--------|---------|:-------:|
| Premier League | England | 500+ |
| La Liga | Spain | 500+ |
| Bundesliga | Germany | 500+ |
| Serie A | Italy | 500+ |
| Ligue 1 | France | 500+ |

**Total: 2,854 players** with 50+ attributes per player

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology |
|----------|------------|
| **GUI Framework** | ![PyQt6](https://img.shields.io/badge/PyQt6-41CD52?style=flat-square&logo=qt&logoColor=white) |
| **Machine Learning** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=python&logoColor=white) |
| **AI Integration** | ![Gemini](https://img.shields.io/badge/Gemini_AI-4285F4?style=flat-square&logo=google&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |

</div>

---

## 📁 Project Structure

```
moneyball_app/
├── 📄 run_app.py           # Application entry point
├── 📄 requirements.txt     # Python dependencies
├── 📁 ai/                  # AI module
│   └── gemini_storyteller.py
├── 📁 data/                # Data files
│   ├── config.json         # API configuration (not tracked)
│   └── dataPlayerC.csv     # Player dataset
├── 📁 gui/                 # GUI components
│   └── main_window.py      # Main application window
├── 📁 ml/                  # Machine Learning
│   ├── preprocessing.py    # Data preprocessing
│   └── similarity_model.py # Similarity algorithms
└── 📁 visualization/       # Chart generation
    └── charts.py
```

<div align="center">

**Made with ❤️ for Football Analytics**

⭐ Star this repo if you find it useful!

</div>
