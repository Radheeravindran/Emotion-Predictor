import pandas as pd
import nltk
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
from datetime import datetime, timedelta

# Download required NLTK resources
nltk.download("punkt")

user_data = [
    {"date": "2025-01-20", "text": "Today is such a beautiful day, I can't stop smiling!"},
    {"date": "2025-01-22", "text": "I can't believe this is happening, I’m so furious right now!"},
    {"date": "2025-01-25", "text": "I feel like everything is slipping through my fingers, and I can't stop it."},
    {"date": "2025-02-20", "text": "This moment feels like pure bliss; I could dance with joy!"},
    {"date": "2025-02-22", "text": "This is completely unfair, I’m done being patient with this situation!"},
    {"date": "2025-02-23", "text": "I feel so lost and alone, like I'm drowning in an ocean of sadness."},
    {"date": "2025-02-24", "text": "I'm so excited for the future, I can't wait to see what it holds!"},
    {"date": "2025-02-25", "text": "i'm afraid of what the future holds, i feel so uncertain."},
    {"date": "2025-02-25", "text": "I just want to curl up and disappear for a while, nothing feels right."}
    
]

df = pd.DataFrame(user_data)
df["date"] = pd.to_datetime(df["date"])

# Step 2: Load Sentiment & Emotion Analysis Model
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Step 3: Perform Emotion Analysis
def analyze_emotion(text):
    result = emotion_pipeline(text)[0][0]
    return result["label"], result["score"]

df["emotion"], df["confidence"] = zip(*df["text"].apply(analyze_emotion))

# Step 4: Plot Emotional Trends Over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="date", y="confidence", hue="emotion", marker="o")
plt.title("Emotional Evolution Over Time")
plt.xlabel("Date")
plt.ylabel("Emotion Confidence")
plt.legend(title="Emotion")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()