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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
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
    "CACHE_TYPE": "SimpleCache",  # Consider using RedisCache for production
    "CACHE_DEFAULT_TIMEOUT": 300  # 5 minutes
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'TestData.csv')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

os.makedirs(STATIC_DIR, exist_ok=True)

# Custom stopwords
# Combining WordCloud's STOPWORDS, NLTK's stopwords, and additional custom words
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
    'also', 'maybe', 'somehow', 'someone', 'somebody', 'everyone', 'need', 'help'
    'anything', 'something', 'everything', 'every', 'ive', 'apartments', ''
    'property', 'properties', 'house', 'houses', 'home', 'homes',
    'any', 'did', 'with', 'stuff', 'the', 'and', 'to', 'is', 'in', 'it', 'you', 'that', 'of', 'things', 'us'
}

# Final stopwords set
custom_stopwords = wordcloud_stopwords.union(nltk_stopwords).union(additional_stopwords)

# =========================
# Data Loading and Preprocessing
# =========================

def clean_text(text):
    """
    Cleans the input text by lowering case, removing non-alphabetic characters,
    and handling contractions.
    """
    text = str(text).lower()
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

# Preprocess reviews at startup
try:
    logger.info("Preprocessing reviews at startup.")
    df_reviews['cleaned_comment'] = df_reviews['comment'].apply(clean_text)
    logger.info("Preprocessing completed.")
except Exception as e:
    logger.error(f"Error during preprocessing: {e}")

# =========================
# WordAnalysis Classes
# =========================

class WordAnalysis:
    """
    Class to store and manage contexts for words.
    """
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

# Separate instances for TextBlob and RoBERTa
word_analyzer_textblob = WordAnalysis()
word_analyzer_roberta = WordAnalysis()

# =========================
# TextBlob Sentiment Analysis
# =========================

def get_sentiment_textblob(text, rating=None):
    """
    Computes sentiment using TextBlob and adjusts it based on critical terms and rating.
    """
    text = clean_text(text)
    analysis = TextBlob(text)

    # Get base polarity
    polarity = analysis.sentiment.polarity

    # Adjust for critical issues (safety, health, legal)
    critical_terms = {
        'shooting': -0.8,
        'scam': -0.7,
        'unsafe': -0.7,
        'roach': -0.6,
        'broke': -0.5,
        'poop': -0.5,
        'trash': -0.4,
        'homeless': -0.4,
        'broken': -0.4,
        'rude': -0.3
    }

    for term, impact in critical_terms.items():
        if term in text.lower():
            polarity += impact

    # If we have a rating, factor it in heavily
    if rating is not None:
        rating_polarity = (rating - 3) / 2  # Convert 1-5 scale to -1 to 1
        polarity = (polarity + rating_polarity * 2) / 3  # Weight rating more heavily

    # Strengthen negative sentiments
    if polarity < 0:
        polarity *= 1.5

    # Scale [-1..+1] to [20..100]
    score = (polarity + 1) * 40 + 20

    return min(max(score, 20), 100), polarity

def compute_mood_score_textblob(days_back, filtered_df, timestamp=None):
    """
    Computes mood score and generates word cloud using TextBlob sentiment analysis.
    """
    cutoff_date = datetime.now() - timedelta(days=days_back)
    period_df = filtered_df[filtered_df["created_at"] >= cutoff_date]

    # Generate cache ID based on filter parameters
    cache_id = hashlib.md5(f"textblob-{days_back}-{filtered_df.shape[0]}-{cutoff_date}".encode()).hexdigest()

    # Reset word analysis if this is a new query
    if word_analyzer_textblob.cache_id != cache_id:
        word_analyzer_textblob.reset()
        word_analyzer_textblob.cache_id = cache_id

    if period_df.empty:
        return 0, f"wordcloud_textblob_{days_back}_EMPTY.png", [], {}

    sentiments = []
    word_sentiments = {}
    word_counts = Counter()

    # Process each review
    for idx, row in period_df.iterrows():
        comment = row['comment']
        score, polarity = get_sentiment_textblob(comment, row.get('rating'))
        sentiments.append(score)

        words = row['cleaned_comment'].split()

        # Filter out stopwords and short words
        filtered_words = [w for w in words if w not in custom_stopwords and len(w) > 2]

        # Update word counts with filtered words
        word_counts.update(filtered_words)

        unique_words = set(filtered_words)  # Iterate over unique words to avoid duplicate contexts
        for w in unique_words:
            if w not in word_sentiments:
                word_sentiments[w] = {'contexts': [], 'polarities': []}
            word_sentiments[w]['contexts'].append(comment)
            word_sentiments[w]['polarities'].append(polarity)

            # Store context in word analyzer
            word_analyzer_textblob.store_context(
                w, comment, polarity,
                row.get('rating', 3),
                row['created_at']
            )

    # Calculate final sentiment for each word
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

    # Calculate overall metrics
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

    text_for_cloud = " ".join(filtered_df["cleaned_comment"].astype(str).tolist())
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=custom_stopwords,
        color_func=enhanced_color_func,
        max_words=100,
        collocations=False
    ).generate(text_for_cloud)

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    wc_filename = f"wordcloud_textblob_{days_back}_{timestamp}.png"
    wc_path = os.path.join(STATIC_DIR, wc_filename)
    wc.to_file(wc_path)

    return final_score, wc_filename, word_counts.most_common(10), word_final_sentiments

# =========================
# RoBERTa Sentiment Analysis
# =========================

# Initialize model and tokenizer once at startup
model_roberta = None
tokenizer_roberta = None
device_roberta = None

def initialize_roberta_model():
    """
    Initializes the RoBERTa model and tokenizer using a pre-trained sentiment analysis model.
    Utilizes DataParallel for parallel processing across multiple GPUs.
    """
    global model_roberta, tokenizer_roberta, device_roberta
    MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"  # Pre-trained sentiment model
    try:
        logger.info(f"Loading RoBERTa model: {MODEL_NAME}")
        tokenizer_roberta = AutoTokenizer.from_pretrained(MODEL_NAME)
        model_roberta = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model_roberta.eval()  # Set model to evaluation mode

        if torch.cuda.is_available():
            device_roberta = torch.device("cuda")
            model_roberta = nn.DataParallel(model_roberta)
            model_roberta.to(device_roberta)
            logger.info(f"RoBERTa model moved to GPU: {torch.cuda.get_device_name(0)}")
        else:
            device_roberta = torch.device("cpu")
            logger.info("RoBERTa model loaded on CPU.")
    except Exception as e:
        logger.error(f"Error loading RoBERTa model: {e}")
        tokenizer_roberta = None
        model_roberta = None
        device_roberta = torch.device("cpu")

def roberta_sentiment_batch(texts, batch_size=32):
    """
    Computes sentiments for a batch of texts using the pre-loaded RoBERTa model.
    Processes texts in batches to optimize GPU/CPU utilization.
    """
    if not model_roberta or not tokenizer_roberta:
        logger.error("RoBERTa model or tokenizer not loaded.")
        return [{"label": "error", "score": 50, "raw_scores": []} for _ in texts]

    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            inputs = tokenizer_roberta(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(device_roberta) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model_roberta(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

            # Get labels and scores
            labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
            for prob in probs:
                label = labels[np.argmax(prob)]
                score = int(prob[np.argmax(prob)] * 100)
                raw_scores = list(prob * 100)
                results.append({
                    "label": label,
                    "score": score,
                    "raw_scores": raw_scores
                })
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            results.extend([{"label": "error", "score": 50, "raw_scores": []} for _ in batch_texts])

    return results

def compute_mood_score_roberta(days_back, filtered_df, timestamp=None):
    """
    Computes mood score and generates word cloud using RoBERTa sentiment analysis.
    Utilizes torch's DataParallel for efficient batch processing.
    """
    cutoff_date = datetime.now() - timedelta(days=days_back)
    period_df = filtered_df[filtered_df["created_at"] >= cutoff_date]

    # Generate cache ID based on filter parameters
    cache_id = hashlib.md5(f"roberta-{days_back}-{filtered_df.shape[0]}-{cutoff_date}".encode()).hexdigest()

    # Reset word analysis if this is a new query
    if word_analyzer_roberta.cache_id != cache_id:
        word_analyzer_roberta.reset()
        word_analyzer_roberta.cache_id = cache_id

    if period_df.empty:
        return 0, f"wordcloud_roberta_{days_back}_EMPTY.png", [], {}

    sentiments = []
    word_sentiments = {}
    word_counts = Counter()

    comments = period_df['cleaned_comment'].tolist()
    ratings = period_df.get('rating', [3] * len(period_df)).tolist()

    # Batch sentiment analysis
    sentiment_results = roberta_sentiment_batch(comments, batch_size=64)

    for comment, sentiment_result, rating in zip(comments, sentiment_results, ratings):
        score = sentiment_result['score']
        # Assuming '1 star' is most negative and '5 stars' is most positive
        sentiment_mapping = {
            "1 star": -1.0,
            "2 stars": -0.5,
            "3 stars": 0.0,
            "4 stars": 0.5,
            "5 stars": 1.0,
            "error": 0.0
        }
        polarity = sentiment_mapping.get(sentiment_result['label'], 0.0)
        sentiments.append(score)

        words = comment.split()

        # Filter out stopwords and short words
        filtered_words = [w for w in words if w not in custom_stopwords and len(w) > 2]

        # Update word counts with filtered words
        word_counts.update(filtered_words)

        unique_words = set(filtered_words)  # Iterate over unique words to avoid duplicate contexts
        for w in unique_words:
            if w not in word_sentiments:
                word_sentiments[w] = {'contexts': [], 'polarities': []}
            word_sentiments[w]['contexts'].append(comment)
            word_sentiments[w]['polarities'].append(polarity)

            # Store context in word analyzer
            word_analyzer_roberta.store_context(
                w, comment, polarity,
                rating,
                period_df.loc[period_df['cleaned_comment'] == comment, 'created_at'].iloc[0]
            )

    # Calculate final sentiment for each word
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

    # Calculate overall metrics
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

    text_for_cloud = " ".join(filtered_df["cleaned_comment"].astype(str).tolist())
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=custom_stopwords,
        color_func=enhanced_color_func,
        max_words=100,
        collocations=False
    ).generate(text_for_cloud)

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    wc_filename = f"wordcloud_roberta_{days_back}_{timestamp}.png"
    wc_path = os.path.join(STATIC_DIR, wc_filename)
    wc.to_file(wc_path)

    return final_score, wc_filename, word_counts.most_common(10), word_final_sentiments

# =========================
# Routes for TextBlob Sentiment
# =========================

@app.route("/")
def index():
    """
    Renders the TextBlob sentiment dashboard.
    """
    logger.info("Accessing TextBlob dashboard.")
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    locations = sorted(df_reviews["location_name"].unique())

    mood_30, wc_30, top_words_30, sentiments_30 = compute_mood_score_textblob(30, df_reviews, timestamp)
    mood_60, wc_60, top_words_60, sentiments_60 = compute_mood_score_textblob(60, df_reviews, timestamp)
    mood_90, wc_90, top_words_90, sentiments_90 = compute_mood_score_textblob(90, df_reviews, timestamp)
    mood_365, wc_365, top_words_365, sentiments_365 = compute_mood_score_textblob(365, df_reviews, timestamp)

    return render_template(
        "index.html",
        locations=locations,
        mood_30=mood_30, wc_30=wc_30, top_words_30=top_words_30,
        mood_60=mood_60, wc_60=wc_60, top_words_60=top_words_60,
        mood_90=mood_90, wc_90=wc_90, top_words_90=top_words_90,
        mood_365=mood_365, wc_365=wc_365, top_words_365=top_words_365,
        word_sentiments_30=sentiments_30,
        word_sentiments_60=sentiments_60,
        word_sentiments_90=sentiments_90,
        word_sentiments_365=sentiments_365,
        now=timestamp,
        method='textblob'
    )

@app.route("/filter", methods=["GET"])
def filter_data():
    """
    Filters data based on location and sentiment analysis method.
    """
    location = request.args.get('location', '')
    method = request.args.get('method', 'textblob')  # 'textblob' or 'roberta'
    logger.info(f"Filter request received for location: {location} using method: {method}")

    if method == 'textblob':
        word_analyzer = word_analyzer_textblob
        compute_mood_score = compute_mood_score_textblob
    elif method == 'roberta':
        word_analyzer = word_analyzer_roberta
        compute_mood_score = compute_mood_score_roberta
    else:
        logger.warning(f"Unknown method '{method}' specified. Defaulting to 'textblob'.")
        word_analyzer = word_analyzer_textblob
        compute_mood_score = compute_mood_score_textblob

    if location:
        filtered_df = df_reviews[df_reviews['location_name'] == location].copy()
    else:
        filtered_df = df_reviews.copy()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    mood_30, wc_30, top_words_30, sentiments_30 = compute_mood_score(30, filtered_df, timestamp)
    mood_60, wc_60, top_words_60, sentiments_60 = compute_mood_score(60, filtered_df, timestamp)
    mood_90, wc_90, top_words_90, sentiments_90 = compute_mood_score(90, filtered_df, timestamp)
    mood_365, wc_365, top_words_365, sentiments_365 = compute_mood_score(365, filtered_df, timestamp)

    return jsonify({
        'mood_30': mood_30, 'wc_30': wc_30, 'top_words_30': top_words_30,
        'mood_60': mood_60, 'wc_60': wc_60, 'top_words_60': top_words_60,
        'mood_90': mood_90, 'wc_90': wc_90, 'top_words_90': top_words_90,
        'mood_365': mood_365, 'wc_365': wc_365, 'top_words_365': top_words_365,
        'word_sentiments_30': sentiments_30,
        'word_sentiments_60': sentiments_60,
        'word_sentiments_90': sentiments_90,
        'word_sentiments_365': sentiments_365
    })

@app.route("/word_analysis/<word>", methods=["GET"])
def word_analysis(word):
    """
    Retrieves detailed analysis for a specific word.
    """
    # Determine which word analyzer to use based on query parameter
    method = request.args.get('method', 'textblob')
    if method == 'textblob':
        word_analyzer = word_analyzer_textblob
    elif method == 'roberta':
        word_analyzer = word_analyzer_roberta
    else:
        logger.warning(f"Unknown method '{method}' specified for word_analysis. Defaulting to 'textblob'.")
        word_analyzer = word_analyzer_textblob

    contexts = word_analyzer.get_contexts(word)

    # Sort contexts by sentiment (most negative to most positive)
    contexts.sort(key=lambda x: x['sentiment'])

    return jsonify({
        'word': word,
        'contexts': contexts,
        'total_appearances': len(contexts),
        'positive_contexts': sum(1 for c in contexts if c['sentiment'] > 0),
        'negative_contexts': sum(1 for c in contexts if c['sentiment'] < 0)
    })

# =========================
# Routes for RoBERTa Sentiment
# =========================

@app.route("/roberta")
def roberta_dashboard():
    """
    Renders the RoBERTa sentiment dashboard.
    """
    logger.info("Accessing RoBERTa dashboard.")
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    locations = sorted(df_reviews["location_name"].unique())

    mood_30, wc_30, top_words_30, sentiments_30 = compute_mood_score_roberta(30, df_reviews, timestamp)
    mood_60, wc_60, top_words_60, sentiments_60 = compute_mood_score_roberta(60, df_reviews, timestamp)
    mood_90, wc_90, top_words_90, sentiments_90 = compute_mood_score_roberta(90, df_reviews, timestamp)
    mood_365, wc_365, top_words_365, sentiments_365 = compute_mood_score_roberta(365, df_reviews, timestamp)

    return render_template(
        "roberta_index.html",
        locations=locations,
        mood_30=mood_30, wc_30=wc_30, top_words_30=top_words_30,
        mood_60=mood_60, wc_60=wc_60, top_words_60=top_words_60,
        mood_90=mood_90, wc_90=wc_90, top_words_90=top_words_90,
        mood_365=mood_365, wc_365=wc_365, top_words_365=top_words_365,
        word_sentiments_30=sentiments_30,
        word_sentiments_60=sentiments_60,
        word_sentiments_90=sentiments_90,
        word_sentiments_365=sentiments_365,
        now=timestamp,
        method='roberta'
    )

@app.route("/roberta-filter", methods=["GET"])
def roberta_filter_data():
    """
    Filters data based on location for RoBERTa sentiment analysis.
    """
    location = request.args.get('location', '')
    logger.info(f"RoBERTa Filter request received for location: {location}")

    if location:
        filtered_df = df_reviews[df_reviews['location_name'] == location].copy()
    else:
        filtered_df = df_reviews.copy()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    mood_30, wc_30, top_words_30, sentiments_30 = compute_mood_score_roberta(30, filtered_df, timestamp)
    mood_60, wc_60, top_words_60, sentiments_60 = compute_mood_score_roberta(60, filtered_df, timestamp)
    mood_90, wc_90, top_words_90, sentiments_90 = compute_mood_score_roberta(90, filtered_df, timestamp)
    mood_365, wc_365, top_words_365, sentiments_365 = compute_mood_score_roberta(365, filtered_df, timestamp)

    return jsonify({
        'mood_30': mood_30,
        'mood_60': mood_60,
        'mood_90': mood_90,
        'mood_365': mood_365,
        'wc_30': wc_30,
        'wc_60': wc_60,
        'wc_90': wc_90,
        'wc_365': wc_365,
        'top_words_30': top_words_30,  # list of [word, count]
        'top_words_60': top_words_60,
        'top_words_90': top_words_90,
        'top_words_365': top_words_365,
        'word_sentiments_30': sentiments_30,
        'word_sentiments_60': sentiments_60,
        'word_sentiments_90': sentiments_90,
        'word_sentiments_365': sentiments_365
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

        sentiment_result = roberta_sentiment_batch([text])[0]
        return jsonify(sentiment_result)
    except Exception as e:
        logger.error(f"Error in /roberta-sentiment route: {e}")
        return jsonify({"error": "An error occurred processing the sentiment analysis."}), 500

@app.route("/test-roberta")
def test_roberta():
    """
    Test route to verify RoBERTa sentiment analysis.
    """
    sample_text_positive = "I absolutely love this place! It's fantastic."
    sample_text_negative = "I hate this place. It's terrible."
    sample_text_neutral = "The place is okay, nothing special."
    try:
        positive_result = roberta_sentiment_batch([sample_text_positive])[0]
        negative_result = roberta_sentiment_batch([sample_text_negative])[0]
        neutral_result = roberta_sentiment_batch([sample_text_neutral])[0]
        return jsonify({
            "RoBERTa_Positive": positive_result,
            "RoBERTa_Negative": negative_result,
            "RoBERTa_Neutral": neutral_result,
        })
    except Exception as e:
        logger.error(f"Error in /test-roberta route: {e}")
        return jsonify({"error": "An error occurred during testing."}), 500

# =========================
# Routes for Word Analysis
# =========================

@app.route("/word_analysis_roberta/<word>", methods=["GET"])
def word_analysis_roberta(word):
    """
    Retrieves detailed analysis for a specific word using RoBERTa.
    """
    try:
        contexts = word_analyzer_roberta.get_contexts(word)

        # Sort contexts by sentiment (most negative to most positive)
        contexts.sort(key=lambda x: x['sentiment'])

        return jsonify({
            'word': word,
            'contexts': contexts,
            'total_appearances': len(contexts),
            'positive_contexts': sum(1 for c in contexts if c['sentiment'] > 0),
            'negative_contexts': sum(1 for c in contexts if c['sentiment'] < 0)
        })
    except Exception as e:
        logger.error(f"Error in /word_analysis_roberta/<word> route: {e}")
        return jsonify({"error": "An error occurred processing the word analysis."}), 500

# =========================
# Optional: Cleanup Old Word Clouds
# =========================

def clean_old_wordclouds():
    """
    Removes old word cloud images from the static directory.
    """
    files = glob.glob(os.path.join(STATIC_DIR, "wordcloud_*.png"))
    for f in files:
        try:
            os.remove(f)
            logger.info(f"Removed old wordcloud file: {f}")
        except Exception as e:
            logger.error(f"Error removing file {f}: {e}")

# Uncomment the following lines if you want to periodically clean old word clouds
# You can integrate this with a scheduler like APScheduler if needed
# clean_old_wordclouds()

# =========================
# Run the Application
# =========================
try:
    initialize_roberta_model()
    logger.info("RoBERTa model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RoBERTa model: {e}")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
