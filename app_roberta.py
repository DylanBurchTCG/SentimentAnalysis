from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import os
import glob
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

app = Flask(__name__)

# =========================
# Configuration and Setup
# =========================

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'TestData.csv')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

os.makedirs(STATIC_DIR, exist_ok=True)

# Custom stopwords
custom_stopwords = STOPWORDS.union({
    'will', 'one', 'thing', 'told', 'call', 'got', 'apartment',
    't', 've', 's', 'll', 'd', 'm', 're',
    'didn', 'won', 'don', 'cant', 'wasnt', 'hadnt',
    'day', 'today', 'yesterday', 'month', 'year',
    'get', 'got', 'going', 'went', 'come', 'came',
    'place', 'u', 'great', 'office', 'made', 'even', 'it',

    # Additional filler or common words
    'like', 'really', 'very', 'always', 'still', 'never',
    'also', 'maybe', 'somehow', 'someone', 'somebody', 'everyone',
    'anything', 'something', 'everything', 'every',
    'property', 'properties', 'house', 'houses', 'home', 'homes',
    'any', 'did', 'with', 'stuff'
})

# =========================
# Data Loading
# =========================

def clean_text(text):
    if not isinstance(text, str):
        text = ''
    text = text.lower()
    text = re.sub(r"'t", " not", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

try:
    logger.info(f"Attempting to load data from: {DATA_PATH}")
    df_reviews = pd.read_csv(DATA_PATH)
    df_reviews['created_at'] = pd.to_datetime(df_reviews['created_at'])
    logger.info(f"Successfully loaded {len(df_reviews)} reviews")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    df_reviews = pd.DataFrame(columns=["location_name", "comment", "rating", "created_at"])

# =========================
# TextBlob Sentiment Analysis
# =========================

def get_sentiment(text):
    logger.info("Using TextBlob for sentiment analysis.")
    text = clean_text(text)
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity < 0:
        polarity *= 1.5
    # Scale [-1..+1] to [20..100]
    score = (polarity + 1) * 40 + 20
    return min(max(score, 20), 100), polarity

def generate_colored_wordcloud(text, sentiment_dict):
    """Generate word cloud with colors based on sentiment"""

    def color_func(word, **kwargs):
        sentiment = sentiment_dict.get(word.lower(), 0)
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

def compute_mood_score_for_period_filtered(days_back, filtered_df, timestamp=None, sentiment_analysis='textblob'):
    cutoff_date = datetime.now() - timedelta(days=days_back)
    period_df = filtered_df[filtered_df["created_at"] >= cutoff_date]

    # If empty, return placeholders
    if period_df.empty:
        logger.warning(f"No reviews found for the last {days_back} days.")
        return 0, f"wordcloud_{days_back}_EMPTY.png", [], {}

    sentiments = []
    word_sentiments = {}
    word_counts = Counter()

    for comment in period_df["comment"]:
        if sentiment_analysis == 'textblob':
            score, polarity = get_sentiment(comment)
        elif sentiment_analysis == 'roberta':
            # Ensure comment is a string
            if not isinstance(comment, str):
                logger.warning(f"Non-string comment encountered: {comment}. Setting to empty string.")
                comment = ''
            sentiment_result = roberta_sentiment(comment)
            polarity = sentiment_result['score']  # Mapped to 20, 60, 100
            score = sentiment_result['score']  # Already scaled to [20, 100]
        else:
            # Default to TextBlob
            score, polarity = get_sentiment(comment)

        sentiments.append(score)
        words = str(comment).lower().split()
        for w in words:
            if w not in custom_stopwords and len(w) > 2:
                word_sentiments.setdefault(w, []).append(polarity)
                word_counts[w] += 1

    # Average polarity per word
    for w in word_sentiments:
        word_sentiments[w] = sum(word_sentiments[w]) / len(word_sentiments[w])

    mood_score = sum(sentiments) / len(sentiments)
    avg_rating = period_df["rating"].mean() if "rating" in period_df else 3  # fallback
    rating_score = (avg_rating / 5) * 100
    final_score = (mood_score + rating_score) / 2

    # Generate word cloud
    text_for_cloud = " ".join(period_df["comment"].astype(str).tolist())
    wc = generate_colored_wordcloud(text_for_cloud, word_sentiments)

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    wc_filename = f"wordcloud_{days_back}_{timestamp}.png"
    wc_path = os.path.join(STATIC_DIR, wc_filename)
    try:
        wc.to_file(wc_path)
    except Exception as e:
        logger.error(f"Error saving wordcloud image: {e}")
        wc_filename = f"wordcloud_{days_back}_error.png"

    # Top 10 words by frequency
    top_words = word_counts.most_common(10)

    return final_score, wc_filename, top_words, word_sentiments

# =========================
# RoBERTa Sentiment Analysis
# =========================

# Load RoBERTa model and tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Replace with your chosen model
try:
    logger.info(f"Loading RoBERTa model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()  # Set model to evaluation mode
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("RoBERTa model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading RoBERTa model: {e}")
    tokenizer = None
    model = None
    device = torch.device("cpu")

MAX_LENGTH = 512  # Adjust based on your model's requirements

def chunk_text(text, chunk_size=512, overlap=50):
    """Splits text into chunks of size `chunk_size`, with `overlap` tokens."""
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        start += (chunk_size - overlap)
    return chunks

def roberta_sentiment(text):
    """Computes sentiment using RoBERTa model. Returns label and mapped score."""
    try:
        if not tokenizer or not model:
            logger.error("RoBERTa model or tokenizer not loaded. Returning error sentiment.")
            return {"label": "error", "score": 50, "raw_scores": []}  # Neutral default

        logger.info("Using RoBERTa for sentiment analysis.")
        if not isinstance(text, str):
            logger.warning(f"Non-string text received: {text}. Setting to empty string.")
            text = ''
        text = text.strip()
        if not text:
            return {"label": "neutral", "score": 60, "raw_scores": []}

        # Chunk the text if too long
        tokenized_text = tokenizer.tokenize(text)
        if len(tokenized_text) > MAX_LENGTH:
            text_chunks = chunk_text(text, chunk_size=MAX_LENGTH, overlap=50)
        else:
            text_chunks = [text]

        all_labels = []
        for chunk in text_chunks:
            inputs = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
            label_idx = probs.argmax().item()
            label = ["negative", "neutral", "positive"][label_idx]
            all_labels.append(label)

        # Majority vote for label
        label_counts = Counter(all_labels)
        majority_label = label_counts.most_common(1)[0][0]

        # Map labels to scores
        label_to_score = {
            "negative": 20,
            "neutral": 60,
            "positive": 100
        }
        score = label_to_score.get(majority_label, 60)  # Default to neutral

        logger.info(f"RoBERTa sentiment: {majority_label} with score {score}")

        return {
            "label": majority_label,
            "score": score,
            "raw_scores": [label_to_score.get(label, 60) for label in all_labels]
        }
    except Exception as e:
        logger.error(f"Error in roberta_sentiment: {e}")
        return {"label": "error", "score": 50, "raw_scores": []}  # Neutral default

# =========================
# Routes for TextBlob Sentiment
# =========================

@app.route("/")
def index():
    logger.info("Accessing TextBlob dashboard.")
    # clean_old_wordclouds()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    locations = sorted(df_reviews["location_name"].unique())

    mood_30, wc_30, top_words_30, _ = compute_mood_score_for_period_filtered(30, df_reviews, timestamp, sentiment_analysis='textblob')
    mood_60, wc_60, top_words_60, _ = compute_mood_score_for_period_filtered(60, df_reviews, timestamp, sentiment_analysis='textblob')
    mood_90, wc_90, top_words_90, _ = compute_mood_score_for_period_filtered(90, df_reviews, timestamp, sentiment_analysis='textblob')

    return render_template(
        "index.html",
        locations=locations,
        mood_30=mood_30, wc_30=wc_30, top_words_30=top_words_30,
        mood_60=mood_60, wc_60=wc_60, top_words_60=top_words_60,
        mood_90=mood_90, wc_90=wc_90, top_words_90=top_words_90,
        now=timestamp
    )

@app.route("/filter")
def filter_data():
    location = request.args.get('location', '')
    method = request.args.get('method', 'textblob')  # Default to 'textblob'
    logger.info(f"Filter request received for location: {location} using method: {method}")

    if location:
        filtered_df = df_reviews[df_reviews['location_name'] == location].copy()
    else:
        filtered_df = df_reviews.copy()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    mood_30, wc_30, top_words_30, _ = compute_mood_score_for_period_filtered(30, filtered_df, timestamp, sentiment_analysis=method)
    mood_60, wc_60, top_words_60, _ = compute_mood_score_for_period_filtered(60, filtered_df, timestamp, sentiment_analysis=method)
    mood_90, wc_90, top_words_90, _ = compute_mood_score_for_period_filtered(90, filtered_df, timestamp, sentiment_analysis=method)

    return jsonify({
        'mood_30': mood_30,
        'mood_60': mood_60,
        'mood_90': mood_90,
        'wc_30': wc_30,
        'wc_60': wc_60,
        'wc_90': wc_90,
        'top_words_30': top_words_30,  # list of [word, count]
        'top_words_60': top_words_60,
        'top_words_90': top_words_90
    })

# =========================
# Routes for RoBERTa Sentiment
# =========================

@app.route("/roberta")
def roberta_dashboard():
    logger.info("Accessing RoBERTa dashboard.")
    # Fetch and process data for RoBERTa sentiment
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    locations = sorted(df_reviews["location_name"].unique())

    mood_30, wc_30, top_words_30, _ = compute_mood_score_for_period_filtered(
        30, df_reviews, timestamp, sentiment_analysis='roberta'
    )
    mood_60, wc_60, top_words_60, _ = compute_mood_score_for_period_filtered(
        60, df_reviews, timestamp, sentiment_analysis='roberta'
    )
    mood_90, wc_90, top_words_90, _ = compute_mood_score_for_period_filtered(
        90, df_reviews, timestamp, sentiment_analysis='roberta'
    )

    return render_template(
        "roberta_index.html",
        locations=locations,
        mood_30=mood_30, wc_30=wc_30, top_words_30=top_words_30,
        mood_60=mood_60, wc_60=wc_60, top_words_60=top_words_60,
        mood_90=mood_90, wc_90=wc_90, top_words_90=top_words_90,
        now=timestamp
    )

@app.route("/roberta-filter")
def roberta_filter_data():
    location = request.args.get('location', '')
    logger.info(f"RoBERTa Filter request received for location: {location}")

    if location:
        filtered_df = df_reviews[df_reviews['location_name'] == location].copy()
    else:
        filtered_df = df_reviews.copy()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    mood_30, wc_30, top_words_30, _ = compute_mood_score_for_period_filtered(
        30, filtered_df, timestamp, sentiment_analysis='roberta'
    )
    mood_60, wc_60, top_words_60, _ = compute_mood_score_for_period_filtered(
        60, filtered_df, timestamp, sentiment_analysis='roberta'
    )
    mood_90, wc_90, top_words_90, _ = compute_mood_score_for_period_filtered(
        90, filtered_df, timestamp, sentiment_analysis='roberta'
    )

    return jsonify({
        'mood_30': mood_30,
        'mood_60': mood_60,
        'mood_90': mood_90,
        'wc_30': wc_30,
        'wc_60': wc_60,
        'wc_90': wc_90,
        'top_words_30': top_words_30,  # list of [word, count]
        'top_words_60': top_words_60,
        'top_words_90': top_words_90
    })

@app.route("/roberta-sentiment", methods=["POST"])
def roberta_sentiment_route():
    """
    API endpoint that accepts JSON payload:
    {
        "text": "Very long text..."
    }
    Returns JSON with RoBERTa-based sentiment classification:
    {
        "label": "positive",
        "score": 100,
        "raw_scores": [20, 60, 100]
    }
    """
    try:
        data = request.get_json()
        if not data or "text" not in data:
            logger.warning("No text provided in request.")
            return jsonify({"error": "No text provided."}), 400

        text = data.get("text", "")
        if not isinstance(text, str):
            logger.warning(f"Non-string text received: {text}. Setting to empty string.")
            text = ''

        sentiment_result = roberta_sentiment(text)
        return jsonify(sentiment_result)
    except Exception as e:
        logger.error(f"Error in /roberta-sentiment route: {e}")
        return jsonify({"error": "An error occurred processing the sentiment analysis."}), 500

# =========================
# Test Route for RoBERTa Sentiment
# =========================

@app.route("/test-roberta")
def test_roberta():
    sample_text_positive = "I absolutely love this place! It's fantastic."
    sample_text_negative = "I hate this place. It's terrible."
    sample_text_neutral = "The place is okay, nothing special."
    return jsonify({
        "RoBERTa_Positive": roberta_sentiment(sample_text_positive),
        "RoBERTa_Negative": roberta_sentiment(sample_text_negative),
        "RoBERTa_Neutral": roberta_sentiment(sample_text_neutral),
    })

# =========================
# Optional: Cleanup Old Word Clouds
# =========================

# Uncomment and use if you want to clean old wordcloud images
# def clean_old_wordclouds():
#     files = glob.glob(os.path.join(STATIC_DIR, "wordcloud_*.png"))
#     for f in files:
#         try:
#             os.remove(f)
#         except Exception as e:
#             logger.error(f"Error removing file {f}: {e}")

# =========================
# Run the Application
# =========================

if __name__ == "__main__":
    app.run(debug=True, port=5001)
