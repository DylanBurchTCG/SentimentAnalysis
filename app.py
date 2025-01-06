from flask import Flask, render_template
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
from wordcloud import WordCloud
import os

app = Flask(__name__)

# Get absolute path to the data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'TestData.csv')

# Create directories if they don't exist
os.makedirs(os.path.join(BASE_DIR, 'static'), exist_ok=True)

# Load Reviews Data
try:
    print(f"Attempting to load data from: {DATA_PATH}")
    df_reviews = pd.read_csv(DATA_PATH)
    # Convert date column to datetime
    df_reviews['created_at'] = pd.to_datetime(df_reviews['created_at'])
    print(f"Successfully loaded {len(df_reviews)} reviews")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    df_reviews = pd.DataFrame(columns=["location_name", "comment", "rating", "created_at"])

def get_sentiment(text):
    """Get sentiment score for a piece of text"""
    analysis = TextBlob(str(text))
    # Convert polarity (-1 to 1) to percentage (0 to 100)
    return (analysis.sentiment.polarity + 1) * 50

def compute_mood_score_for_period(days_back):
    cutoff_date = datetime.now() - timedelta(days=days_back)
    # Use created_at instead of date
    period_df = df_reviews[df_reviews["created_at"] >= cutoff_date]

    if period_df.empty:
        return 0, None

    # Calculate sentiment scores using the comment field
    sentiments = [get_sentiment(text) for text in period_df["comment"]]
    mood_score = sum(sentiments) / len(sentiments)

    # Also factor in the actual ratings (assuming they're on a 1-5 scale)
    avg_rating = period_df["rating"].mean()
    # Convert rating to percentage (assuming 5 is max rating)
    rating_score = (avg_rating / 5) * 100

    # Combine sentiment and rating scores
    final_score = (mood_score + rating_score) / 2

    # Generate WordCloud from comments
    text_for_cloud = " ".join(period_df["comment"].astype(str).tolist())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_for_cloud)

    # Save to static folder
    wc_filename = f"wordcloud_{days_back}.png"
    wc.to_file(os.path.join("static", wc_filename))

    return final_score, wc_filename

@app.route("/")
def index():
    # Compute mood scores and get wordclouds
    mood_30, wc_30 = compute_mood_score_for_period(30)
    mood_60, wc_60 = compute_mood_score_for_period(60)
    mood_90, wc_90 = compute_mood_score_for_period(90)

    return render_template(
        "index.html",
        mood_30=mood_30,
        wc_30=wc_30,
        mood_60=mood_60,
        wc_60=wc_60,
        mood_90=mood_90,
        wc_90=wc_90
    )

if __name__ == "__main__":
    app.run(debug=True)