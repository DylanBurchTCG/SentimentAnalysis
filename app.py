from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
from wordcloud import WordCloud, STOPWORDS
import os
import glob

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'TestData.csv')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Ensure static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

# # If needed, you can re-enable this function to remove old images on each request.
# def clean_old_wordclouds():
#     """Clean up old wordcloud files"""
#     files = glob.glob(os.path.join(STATIC_DIR, "wordcloud_*.png"))
#     for f in files:
#         try:
#             os.remove(f)
#         except:
#             pass

# Custom stopwords
custom_stopwords = STOPWORDS.union({
    'will', 'one', 'thing', 'told', 'call', 'got', 'apartment',
    't', 've', 's', 'll', 'd', 'm', 're',
    'didn', 'won', 'don', 'cant', 'wasnt', 'hadnt',
    'day', 'today', 'yesterday', 'month', 'year',
    'get', 'got', 'going', 'went', 'come', 'came', 'place', 'u'
})

def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r"'t", " not", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Load reviews
try:
    print(f"Attempting to load data from: {DATA_PATH}")
    df_reviews = pd.read_csv(DATA_PATH)
    df_reviews['created_at'] = pd.to_datetime(df_reviews['created_at'])
    print(f"Successfully loaded {len(df_reviews)} reviews")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    df_reviews = pd.DataFrame(columns=["location_name", "comment", "rating", "created_at"])

def get_sentiment(text):
    text = clean_text(text)
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    # Slight penalty for negative polarity
    if polarity < 0:
        polarity = polarity * 1.5

    # Scale polarity into [20,100]
    score = (polarity + 1) * 40 + 20
    return min(max(score, 20), 100), polarity

def generate_colored_wordcloud(text, sentiment_dict):
    """Generate word cloud with colors based on sentiment"""
    def color_func(word, **kwargs):
        sentiment = sentiment_dict.get(word.lower(), 0)
        # Green for positive, red for negative, adjust brightness
        hue = 120 if sentiment >= 0 else 0
        lightness = max(20, min(80, 50 + sentiment * 30))
        return f"hsl({hue}, 50%, {lightness}%)"

    return WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=custom_stopwords,
        color_func=color_func,
        max_words=100,
        collocations=False
    ).generate(text)

def compute_mood_score_for_period_filtered(days_back, filtered_df, timestamp=None):
    """Compute mood for last X days in a filtered dataframe, generate word cloud."""
    cutoff_date = datetime.now() - timedelta(days=days_back)
    period_df = filtered_df[filtered_df["created_at"] >= cutoff_date]

    # If no reviews, return 0 with a placeholder filename
    if period_df.empty:
        return 0, f"wordcloud_{days_back}_EMPTY.png", {}

    sentiments = []
    word_sentiments = {}

    for comment in period_df["comment"]:
        score, polarity = get_sentiment(comment)
        sentiments.append(score)
        words = str(comment).lower().split()
        for word in words:
            if word not in custom_stopwords and len(word) > 2:
                if word not in word_sentiments:
                    word_sentiments[word] = []
                word_sentiments[word].append(polarity)

    word_sentiments = {
        w: sum(vals) / len(vals)
        for w, vals in word_sentiments.items()
    }

    mood_score = sum(sentiments) / len(sentiments)
    avg_rating = period_df["rating"].mean()
    rating_score = (avg_rating / 5) * 100
    final_score = (mood_score + rating_score) / 2

    # Build the wordcloud
    text_for_cloud = " ".join(period_df["comment"].astype(str).tolist())
    wc = generate_colored_wordcloud(text_for_cloud, word_sentiments)

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    wc_filename = f"wordcloud_{days_back}_{timestamp}.png"
    wc_path = os.path.join(STATIC_DIR, wc_filename)
    wc.to_file(wc_path)

    return final_score, wc_filename, word_sentiments

@app.route("/")
def index():
    # clean_old_wordclouds()  # Optional cleanup if desired
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    locations = sorted(df_reviews["location_name"].unique())

    mood_30, wc_30, _ = compute_mood_score_for_period_filtered(30, df_reviews, timestamp)
    mood_60, wc_60, _ = compute_mood_score_for_period_filtered(60, df_reviews, timestamp)
    mood_90, wc_90, _ = compute_mood_score_for_period_filtered(90, df_reviews, timestamp)

    return render_template(
        "index.html",
        locations=locations,
        mood_30=mood_30, wc_30=wc_30,
        mood_60=mood_60, wc_60=wc_60,
        mood_90=mood_90, wc_90=wc_90,
        now=timestamp
    )

@app.route("/filter")
def filter_data():
    location = request.args.get('location', '')
    # clean_old_wordclouds()  # Optional cleanup if desired
    print(f"Filter request received for location: {location}")

    if location:
        filtered_df = df_reviews[df_reviews['location_name'] == location].copy()
    else:
        filtered_df = df_reviews.copy()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    mood_30, wc_30, _ = compute_mood_score_for_period_filtered(30, filtered_df, timestamp)
    mood_60, wc_60, _ = compute_mood_score_for_period_filtered(60, filtered_df, timestamp)
    mood_90, wc_90, _ = compute_mood_score_for_period_filtered(90, filtered_df, timestamp)

    print(f"Generated new wordclouds: {wc_30}, {wc_60}, {wc_90}")

    return jsonify({
        'mood_30': float(mood_30),
        'mood_60': float(mood_60),
        'mood_90': float(mood_90),
        'wc_30': wc_30,
        'wc_60': wc_60,
        'wc_90': wc_90
    })

if __name__ == "__main__":
    app.run(debug=True)
