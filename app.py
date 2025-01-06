from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
from wordcloud import WordCloud, STOPWORDS
import os

app = Flask(__name__)

# Get absolute path to the data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'TestData.csv')

# Create directories if they don't exist
os.makedirs(os.path.join(BASE_DIR, 'static'), exist_ok=True)

# Add custom stop words
custom_stopwords = STOPWORDS.union({'will', 'one', 'thing', 'told', 'call', 'got', 'apartment'})

# Load Reviews Data
try:
    print(f"Attempting to load data from: {DATA_PATH}")
    df_reviews = pd.read_csv(DATA_PATH)
    df_reviews['created_at'] = pd.to_datetime(df_reviews['created_at'])
    print(f"Successfully loaded {len(df_reviews)} reviews")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    df_reviews = pd.DataFrame(columns=["location_name", "comment", "rating", "created_at"])


def get_sentiment(text):
    """Get sentiment score and polarity for a piece of text"""
    analysis = TextBlob(str(text))
    # Return both percentage and raw polarity
    return (analysis.sentiment.polarity + 1) * 50, analysis.sentiment.polarity


def generate_colored_wordcloud(text, sentiment_dict):
    """Generate word cloud with colors based on sentiment"""

    def color_func(word, **kwargs):
        # Get sentiment for the word, default to neutral
        sentiment = sentiment_dict.get(word.lower(), 0)
        # Red for negative, green for positive
        return f"hsl({120 if sentiment >= 0 else 0}, 50%, {max(20, min(80, 50 + sentiment * 30))}%)"

    return WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=custom_stopwords,
        color_func=color_func,
        max_words=100,
        collocations=False
    ).generate(text)


def compute_mood_score_for_period(days_back):
    cutoff_date = datetime.now() - timedelta(days=days_back)
    period_df = df_reviews[df_reviews["created_at"] >= cutoff_date]

    if period_df.empty:
        return 0, None, {}

    # Calculate sentiments and build word sentiment dictionary
    word_sentiments = {}
    sentiments = []

    for comment in period_df["comment"]:
        score, polarity = get_sentiment(comment)
        sentiments.append(score)

        # Calculate sentiment for individual words
        words = str(comment).lower().split()
        for word in words:
            if word not in custom_stopwords and len(word) > 2:
                if word not in word_sentiments:
                    word_sentiments[word] = []
                word_sentiments[word].append(polarity)

    # Average the sentiments for each word
    word_sentiments = {word: sum(scores) / len(scores) for word, scores in word_sentiments.items()}

    # Calculate mood score
    mood_score = sum(sentiments) / len(sentiments)

    # Factor in ratings
    avg_rating = period_df["rating"].mean()
    rating_score = (avg_rating / 5) * 100
    final_score = (mood_score + rating_score) / 2

    # Generate improved WordCloud
    text_for_cloud = " ".join(period_df["comment"].astype(str).tolist())
    wc = generate_colored_wordcloud(text_for_cloud, word_sentiments)

    # Save to static folder
    wc_filename = f"wordcloud_{days_back}.png"
    wc.to_file(os.path.join("static", wc_filename))

    return final_score, wc_filename, word_sentiments


@app.route("/")
def index():
    # Get unique locations for dropdown
    locations = sorted(df_reviews["location_name"].unique())

    # Compute mood scores and get wordclouds
    mood_30, wc_30, sentiments_30 = compute_mood_score_for_period(30)
    mood_60, wc_60, sentiments_60 = compute_mood_score_for_period(60)
    mood_90, wc_90, sentiments_90 = compute_mood_score_for_period(90)

    return render_template(
        "index.html",
        locations=locations,
        mood_30=mood_30,
        wc_30=wc_30,
        mood_60=mood_60,
        wc_60=wc_60,
        mood_90=mood_90,
        wc_90=wc_90
    )


@app.route("/filter")
def filter_data():
    location = request.args.get('location', '')

    def get_filtered_mood(days):
        if location:
            filtered_df = df_reviews[df_reviews['location_name'] == location]
        else:
            filtered_df = df_reviews

        score, _, _ = compute_mood_score_for_period_filtered(days, filtered_df)
        return score

    return jsonify({
        'mood_30': get_filtered_mood(30),
        'mood_60': get_filtered_mood(60),
        'mood_90': get_filtered_mood(90)
    })
def compute_mood_score_for_period_filtered(days_back, filtered_df):
    """Version of compute_mood_score_for_period that takes a pre-filtered dataframe"""
    cutoff_date = datetime.now() - timedelta(days=days_back)
    period_df = filtered_df[filtered_df["created_at"] >= cutoff_date]

    if period_df.empty:
        return 0, None, {}

    # Calculate sentiments and build word sentiment dictionary
    word_sentiments = {}
    sentiments = []

    for comment in period_df["comment"]:
        score, polarity = get_sentiment(comment)
        sentiments.append(score)

        # Calculate sentiment for individual words
        words = str(comment).lower().split()
        for word in words:
            if word not in custom_stopwords and len(word) > 2:
                if word not in word_sentiments:
                    word_sentiments[word] = []
                word_sentiments[word].append(polarity)

    # Average the sentiments for each word
    word_sentiments = {word: sum(scores) / len(scores) for word, scores in word_sentiments.items()}

    # Calculate mood score
    mood_score = sum(sentiments) / len(sentiments)

    # Factor in ratings
    avg_rating = period_df["rating"].mean()
    rating_score = (avg_rating / 5) * 100
    final_score = (mood_score + rating_score) / 2

    # Generate improved WordCloud
    text_for_cloud = " ".join(period_df["comment"].astype(str).tolist())
    wc = generate_colored_wordcloud(text_for_cloud, word_sentiments)

    # Save to static folder with location-specific filename
    wc_filename = f"wordcloud_{days_back}.png"
    wc.to_file(os.path.join("static", wc_filename))

    return final_score, wc_filename, word_sentiments
if __name__ == "__main__":
    app.run(debug=True)