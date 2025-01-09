import os
import glob
import hashlib
import re
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Initialize Flask app
app = Flask(__name__)

# =========================
# Configuration and Setup
# =========================

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Flask-Caching
cache_config = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'TestData-2.csv')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

os.makedirs(STATIC_DIR, exist_ok=True)

# Custom stopwords setup
wordcloud_stopwords = set(STOPWORDS)
nltk_stopwords = set(stopwords.words('english'))
additional_stopwords = {
    'will', 'one', 'thing', 'told', 'call', 'got', 'apartment',
    't', 've', 's', 'll', 'd', 'm', 're',
    'didn', 'won', 'don', 'cant', 'wasnt', 'hadnt',
    'day', 'today', 'yesterday', 'month', 'year',
    'get', 'got', 'going', 'went', 'come', 'came',
    'place', 'u', 'great', 'office', 'made', 'even', 'it',
    'like', 'really', 'very', 'always', 'still', 'never',
    'also', 'maybe', 'somehow', 'someone', 'somebody', 'everyone', 'need', 'help',
    'anything', 'something', 'everything', 'every', 'ive', 'apartments',
    'property', 'properties', 'house', 'houses', 'home', 'homes',
    'any', 'did', 'with', 'stuff', 'the', 'and', 'to', 'is', 'in', 'it', 'you', 'that', 'of', 'things', 'us'
}
custom_stopwords = wordcloud_stopwords.union(nltk_stopwords).union(additional_stopwords)


# =========================
# Data Loading and Preprocessing
# =========================

def clean_text(text):
    """Cleans text by lowering case and removing non-alphabetic characters."""
    text = str(text).lower()
    text = re.sub(r"'t", " not", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text


# Load and preprocess data
try:
    logger.info(f"Loading data from: {DATA_PATH}")
    df_reviews = pd.read_csv(DATA_PATH)
    df_reviews['created_at'] = pd.to_datetime(df_reviews['created_at'])
    df_reviews['cleaned_comment'] = df_reviews['comment'].apply(clean_text)
    logger.info(f"Successfully loaded {len(df_reviews)} reviews")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    df_reviews = pd.DataFrame(columns=["location_name", "comment", "rating", "created_at"])


# =========================
# Word Analysis Class
# =========================

class WordAnalysis:
    def __init__(self):
        self.word_contexts = defaultdict(list)
        self.cache_id = None

    def reset(self):
        self.word_contexts.clear()
        self.cache_id = None

    def store_context(self, word, comment, sentiment, rating, date):
        context = {
            'comment': comment,
            'sentiment': sentiment,
            'rating': rating,
            'date': date.strftime('%Y-%m-%d')
        }
        self.word_contexts[word].append(context)

    def get_contexts(self, word):
        return self.word_contexts.get(word, [])


word_analyzer_textblob = WordAnalysis()


# =========================
# Sentiment Analysis Functions
# =========================

def get_sentiment_textblob(text, rating=None):
    """Compute sentiment score using TextBlob with adjustments."""
    text = clean_text(text)
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    # Adjust for critical terms
    critical_terms = {
        'shooting': -0.8, 'scam': -0.7, 'unsafe': -0.7,
        'roach': -0.6, 'broke': -0.5, 'poop': -0.5,
        'trash': -0.4, 'homeless': -0.4, 'broken': -0.4,
        'rude': -0.3
    }

    for term, impact in critical_terms.items():
        if term in text.lower():
            polarity += impact

    # Factor in rating if available
    if rating is not None:
        rating_polarity = (rating - 3) / 2
        polarity = (polarity + rating_polarity * 2) / 3

    # Strengthen negative sentiments
    if polarity < 0:
        polarity *= 1.5

    # Scale to final score
    score = (polarity + 1) * 40 + 20
    return min(max(score, 20), 100), polarity


def compute_mood_score_textblob(days_back, filtered_df, timestamp=None):
    """Compute mood score and generate word cloud."""
    # Initialize data structures
    sentiments = []
    word_sentiments = {}
    word_counts = Counter()
    collected_words = []

    # Set time boundary
    cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_back)
    period_df = filtered_df[filtered_df["created_at"] >= cutoff_date]

    if period_df.empty:
        return 0, f"wordcloud_textblob_{days_back}_EMPTY.png", [], {}

    # Generate cache ID
    cache_id = hashlib.md5(f"textblob-{days_back}-{filtered_df.shape[0]}-{cutoff_date}".encode()).hexdigest()
    if word_analyzer_textblob.cache_id != cache_id:
        word_analyzer_textblob.reset()
        word_analyzer_textblob.cache_id = cache_id

    # Process each review
    for idx, row in period_df.iterrows():
        # Get sentiment
        score, polarity = get_sentiment_textblob(row['comment'], row.get('rating'))
        sentiments.append(score)

        # Process words
        words = row['cleaned_comment'].split()
        filtered_words = [w for w in words if w not in custom_stopwords and len(w) > 2]
        collected_words.extend(filtered_words)
        word_counts.update(filtered_words)

        # Store word contexts
        for w in set(filtered_words):
            if w not in word_sentiments:
                word_sentiments[w] = {'contexts': [], 'polarities': []}
            word_sentiments[w]['contexts'].append(row['comment'])
            word_sentiments[w]['polarities'].append(polarity)

            word_analyzer_textblob.store_context(
                w, row['comment'], polarity,
                row.get('rating', 3),
                row['created_at']
            )

    # Calculate word sentiments
    word_final_sentiments = {}
    for word, data in word_sentiments.items():
        avg_sentiment = sum(data['polarities']) / len(data['polarities'])
        pos_contexts = sum(1 for p in data['polarities'] if p > 0)
        neg_contexts = sum(1 for p in data['polarities'] if p < 0)

        if pos_contexts > 2 * neg_contexts:
            avg_sentiment = min(1.0, avg_sentiment * 1.5)
        elif neg_contexts > 2 * pos_contexts:
            avg_sentiment = max(-1.0, avg_sentiment * 1.5)

        word_final_sentiments[word] = avg_sentiment

    # Calculate final score
    mood_score = sum(sentiments) / len(sentiments)
    avg_rating = period_df["rating"].mean() if "rating" in period_df else 3
    rating_score = (avg_rating / 5) * 100
    final_score = (mood_score + rating_score) / 2

    # Generate word cloud
    def enhanced_color_func(word, **kwargs):
        sentiment = word_final_sentiments.get(word.lower(), 0)
        if sentiment < 0:
            intensity = int(abs(sentiment) * 255)
            return f"rgb({min(255, 150 + intensity)}, 0, 0)"
        else:
            intensity = int(sentiment * 255)
            return f"rgb(0, {min(255, 100 + intensity)}, 0)"

    text_for_cloud = " ".join(collected_words)
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=custom_stopwords,
        color_func=enhanced_color_func,
        max_words=100,
        collocations=False
    ).generate(text_for_cloud)

    # Save word cloud
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    wc_filename = f"wordcloud_textblob_{days_back}_{timestamp}.png"
    wc_path = os.path.join(STATIC_DIR, wc_filename)
    wc.to_file(wc_path)

    return final_score, wc_filename, word_counts.most_common(10), word_final_sentiments


# =========================
# Routes
# =========================

@app.route("/")
def index():
    """Render the main dashboard."""
    logger.info("Accessing dashboard")
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    locations = sorted(df_reviews["location_name"].unique())

    # Calculate metrics for different time periods
    mood_90, wc_90, top_words_90, sentiments_90 = compute_mood_score_textblob(90, df_reviews, timestamp)
    mood_120, wc_120, top_words_120, sentiments_120 = compute_mood_score_textblob(120, df_reviews, timestamp)
    mood_180, wc_180, top_words_180, sentiments_180 = compute_mood_score_textblob(180, df_reviews, timestamp)
    mood_365, wc_365, top_words_365, sentiments_365 = compute_mood_score_textblob(365, df_reviews, timestamp)

    return render_template(
        "index.html",
        locations=locations,
        mood_90=mood_90, wc_90=wc_90, top_words_90=top_words_90,
        mood_120=mood_120, wc_120=wc_120, top_words_120=top_words_120,
        mood_180=mood_180, wc_180=wc_180, top_words_180=top_words_180,
        mood_365=mood_365, wc_365=wc_365, top_words_365=top_words_365,
        word_sentiments_90=sentiments_90,
        word_sentiments_120=sentiments_120,
        word_sentiments_180=sentiments_180,
        word_sentiments_365=sentiments_365,
        now=timestamp,
        method='textblob'
    )


@app.route("/filter", methods=["GET"])
def filter_data():
    """Filter dashboard data by location."""
    location = request.args.get('location', '')
    logger.info(f"Filter request for location: {location}")

    filtered_df = df_reviews[df_reviews['location_name'] == location].copy() if location else df_reviews.copy()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Calculate metrics for filtered data
    mood_90, wc_90, top_words_90, sentiments_90 = compute_mood_score_textblob(90, filtered_df, timestamp)
    mood_120, wc_120, top_words_120, sentiments_120 = compute_mood_score_textblob(120, filtered_df, timestamp)
    mood_180, wc_180, top_words_180, sentiments_180 = compute_mood_score_textblob(180, filtered_df, timestamp)
    mood_365, wc_365, top_words_365, sentiments_365 = compute_mood_score_textblob(365, filtered_df, timestamp)

    return jsonify({
        'mood_90': mood_90, 'wc_90': wc_90, 'top_words_90': top_words_90,
        'mood_120': mood_120, 'wc_120': wc_120, 'top_words_120': top_words_120,
        'mood_180': mood_180, 'wc_180': wc_180, 'top_words_180': top_words_180,
        'mood_365': mood_365, 'wc_365': wc_365, 'top_words_365': top_words_365,
        'word_sentiments_90': sentiments_90,
        'word_sentiments_120': sentiments_120,
        'word_sentiments_180': sentiments_180,
        'word_sentiments_365': sentiments_365
    })


@app.route("/word_analysis/<word>", methods=["GET"])
def word_analysis(word):
    """Get detailed analysis for a specific word."""
    contexts = word_analyzer_textblob.get_contexts(word)
    contexts.sort(key=lambda x: x['sentiment'])

    return jsonify({
        'word': word,
        'contexts': contexts,
        'total_appearances': len(contexts),
        'positive_contexts': sum(1 for c in contexts if c['sentiment'] > 0),
        'negative_contexts': sum(1 for c in contexts if c['sentiment'] < 0)
    })


# =========================
# Utility Functions
# =========================

def clean_old_wordclouds():
    """Remove old word cloud images."""
    files = glob.glob(os.path.join(STATIC_DIR, "wordcloud_*.png"))
    for f in files:
        try:
            os.remove(f)
            logger.info(f"Removed old wordcloud file: {f}")
        except Exception as e:
            logger.error(f"Error removing file {f}: {e}")


def init_app():
    """Initialize the application."""
    # Create static directory if it doesn't exist
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)

    # Clean old wordclouds on startup
    clean_old_wordclouds()

    # Log application start
    logger.info("Application initialized successfully")


# =========================
# Main Entry Point
# =========================

if __name__ == "__main__":
    init_app()
    app.run(debug=True, port=5001)