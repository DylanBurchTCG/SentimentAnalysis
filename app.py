from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
from wordcloud import WordCloud, STOPWORDS
from collections import Counter, defaultdict
import os
import glob
import hashlib
import re

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'TestData.csv')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

os.makedirs(STATIC_DIR, exist_ok=True)

custom_stopwords = STOPWORDS.union({
    'will', 'one', 'thing', 'told', 'call', 'got', 'apartment',
    't', 've', 's', 'll', 'd', 'm', 're',
    'didn', 'won', 'don', 'cant', 'wasnt', 'hadnt',
    'day', 'today', 'yesterday', 'month', 'year',
    'get', 'got', 'going', 'went', 'come', 'came',
    'place', 'u', 'great', 'office', 'made', 'even', 'it',
    'like', 'really', 'very', 'always', 'still', 'never',
    'also', 'maybe', 'somehow', 'someone', 'somebody', 'everyone',
    'anything', 'something', 'everything', 'every',
    'property', 'properties', 'house', 'houses', 'home', 'homes',
    'any', 'did', 'with', 'stuff'
})


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


# Global instance for word analysis
word_analyzer = WordAnalysis()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"'t", " not", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text


def get_sentiment(text):
    text = clean_text(text)
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity < 0:
        polarity *= 1.5
    score = (polarity + 1) * 40 + 20
    return min(max(score, 20), 100), polarity


def compute_mood_score_for_period_filtered(days_back, filtered_df, timestamp=None):
    cutoff_date = datetime.now() - timedelta(days=days_back)
    period_df = filtered_df[filtered_df["created_at"] >= cutoff_date]

    # Generate cache ID based on filter parameters
    cache_id = hashlib.md5(f"{days_back}-{filtered_df.shape[0]}-{cutoff_date}".encode()).hexdigest()

    # Reset word analysis if this is a new query
    if word_analyzer.cache_id != cache_id:
        word_analyzer.reset()
        word_analyzer.cache_id = cache_id

    if period_df.empty:
        return 0, f"wordcloud_{days_back}_EMPTY.png", [], {}

    sentiments = []
    word_sentiments = {}
    word_counts = Counter()

    # Process each review
    for idx, row in period_df.iterrows():
        comment = row['comment']
        score, polarity = get_sentiment(comment)
        sentiments.append(score)

        words = clean_text(comment).split()
        for w in words:
            if w not in custom_stopwords and len(w) > 2:
                if w not in word_sentiments:
                    word_sentiments[w] = {'contexts': [], 'polarities': []}
                word_sentiments[w]['contexts'].append(comment)
                word_sentiments[w]['polarities'].append(polarity)
                word_counts[w] += 1

                # Store context in word analyzer
                word_analyzer.store_context(
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

    text_for_cloud = " ".join(period_df["comment"].astype(str).tolist())
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
    wc_filename = f"wordcloud_{days_back}_{timestamp}.png"
    wc_path = os.path.join(STATIC_DIR, wc_filename)
    wc.to_file(wc_path)

    return final_score, wc_filename, word_counts.most_common(10), word_final_sentiments


# Load data
try:
    print(f"Attempting to load data from: {DATA_PATH}")
    df_reviews = pd.read_csv(DATA_PATH)
    df_reviews['created_at'] = pd.to_datetime(df_reviews['created_at'])
    print(f"Successfully loaded {len(df_reviews)} reviews")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    df_reviews = pd.DataFrame(columns=["location_name", "comment", "rating", "created_at"])


@app.route("/")
def index():
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    locations = sorted(df_reviews["location_name"].unique())
    filtered_df = df_reviews.copy()

    mood_30, wc_30, top_words_30, sentiments_30 = compute_mood_score_for_period_filtered(30, filtered_df, timestamp)
    mood_60, wc_60, top_words_60, sentiments_60 = compute_mood_score_for_period_filtered(60, filtered_df, timestamp)
    mood_90, wc_90, top_words_90, sentiments_90 = compute_mood_score_for_period_filtered(90, filtered_df, timestamp)
    mood_365, wc_365, top_words_365, sentiments_365 = compute_mood_score_for_period_filtered(365, filtered_df,
                                                                                             timestamp)

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
        now=timestamp
    )


@app.route("/filter")
def filter_data():
    location = request.args.get('location', '')
    print(f"Filter request received for location: {location}")

    if location:
        filtered_df = df_reviews[df_reviews['location_name'] == location].copy()
    else:
        filtered_df = df_reviews.copy()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    mood_30, wc_30, top_words_30, sentiments_30 = compute_mood_score_for_period_filtered(30, filtered_df, timestamp)
    mood_60, wc_60, top_words_60, sentiments_60 = compute_mood_score_for_period_filtered(60, filtered_df, timestamp)
    mood_90, wc_90, top_words_90, sentiments_90 = compute_mood_score_for_period_filtered(90, filtered_df, timestamp)
    mood_365, wc_365, top_words_365, sentiments_365 = compute_mood_score_for_period_filtered(365, filtered_df,
                                                                                             timestamp)

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


@app.route("/word_analysis/<word>")
def word_analysis(word):
    """Get detailed analysis for a specific word"""
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


if __name__ == "__main__":
    app.run(debug=True)