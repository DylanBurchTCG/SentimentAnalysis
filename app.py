from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import os
import glob
import hashlib
import re
import logging
import pandas as pd
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)
cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})

# Setup paths and logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'TestData-2.csv')
SUMMARIES_PATH = os.path.join(BASE_DIR, 'data', 'summaries.csv')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
    nltk.download(resource, quiet=True)


class SummaryManager:
    def __init__(self):
        self.summaries_df = self.load_summaries()

    def load_summaries(self):
        """Load property summaries from CSV file."""
        try:
            if not os.path.exists(SUMMARIES_PATH):
                logger.error(f"Summaries file not found at: {SUMMARIES_PATH}")
                return pd.DataFrame()

            df = pd.read_csv(SUMMARIES_PATH)
            df['interval'] = df['interval'].str.extract(r'(\d+)').astype(int)

            if df.empty:
                logger.warning("Loaded summaries file is empty")
            else:
                logger.info(f"Successfully loaded {len(df)} summaries")

            return df
        except Exception as e:
            logger.error(f"Error loading summaries: {str(e)}")
            return pd.DataFrame()

    def get_summary(self, location, period_days):
        """Get summary for specific location and time period."""
        if self.summaries_df.empty:
            return None

        # If no location specified, get summary for all properties combined
        if not location:
            valid_periods = self.summaries_df['interval'].unique().tolist()  # Convert to Python list
            if not valid_periods:
                return None

            closest_period = min(valid_periods, key=lambda x: abs(x - period_days))
            matching_summaries = self.summaries_df[
                self.summaries_df['interval'] == closest_period
                ]

            if len(matching_summaries) == 0:
                return None

            # Aggregate summaries
            all_sentiments = '; '.join(matching_summaries['overall_sentiment'].dropna().astype(str))
            all_troubles = '; '.join(matching_summaries['trouble_points'].dropna().astype(str))
            all_suggestions = '; '.join(matching_summaries['suggestions'].dropna().astype(str))

            return {
                'text': all_sentiments,
                'trouble_points': [point.strip() for point in all_troubles.split(';') if point.strip()],
                'suggestions': [sugg.strip() for sugg in all_suggestions.split(';') if sugg.strip()],
                'actual_period': int(closest_period)  # Convert to Python int
            }

        # Find the closest matching period for specific location
        valid_periods = self.summaries_df[
            self.summaries_df['property_name'] == location
            ]['interval'].unique().tolist()  # Convert to Python list

        if not valid_periods:
            return None

        closest_period = min(valid_periods, key=lambda x: abs(x - period_days))

        # Get the summary row
        try:
            summary = self.summaries_df[
                (self.summaries_df['property_name'] == location) &
                (self.summaries_df['interval'] == closest_period)
                ]
            if len(summary) == 0:
                return None

            # Convert to dict and ensure all values are Python native types
            summary_dict = summary.iloc[0].to_dict()
            return {
                'text': str(summary_dict['overall_sentiment']),
                'trouble_points': [point.strip() for point in str(summary_dict['trouble_points']).split(';') if
                                   point.strip()],
                'suggestions': [sugg.strip() for sugg in str(summary_dict['suggestions']).split(';') if sugg.strip()],
                'actual_period': int(closest_period)  # Convert to Python int
            }
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            return None

        closest_period = min(valid_periods, key=lambda x: abs(x - period_days))

        # Get the summary row
        try:
            summary = self.summaries_df[
                (self.summaries_df['property_name'] == location) &
                (self.summaries_df['interval'] == closest_period)
                ]
            if len(summary) == 0:
                return None
            summary = summary.iloc[0].to_dict()

            return {
                'text': summary['overall_sentiment'],
                'trouble_points': [point.strip() for point in str(summary['trouble_points']).split(';') if
                                   point.strip()],
                'suggestions': [sugg.strip() for sugg in str(summary['suggestions']).split(';') if sugg.strip()],
                'actual_period': closest_period
            }
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            return None


# Initialize summary manager
summary_manager = SummaryManager()


@app.route("/summaries")
def get_summaries():
    """Get summaries for a specific location and period."""
    location = request.args.get('location', '')
    period = request.args.get('period', '90')

    try:
        period_days = int(period)
        summary = summary_manager.get_summary(location, period_days)
        return jsonify({"success": True, "summary": summary})
    except ValueError:
        return jsonify({"success": False, "error": "Invalid period"}), 400


@app.route("/filter")
def filter_data():
    """Filter dashboard data by location."""
    location = request.args.get('location', '')
    filtered_df = df_reviews[df_reviews['location_name'] == location].copy() if location else df_reviews.copy()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    results = {}
    for period in [90, 120, 180, 365]:
        score, wc, words, sentiments = review_analyzer.analyze_period(period, filtered_df, timestamp)
        results[f'mood_{period}'] = score
        results[f'wc_{period}'] = wc
        results[f'top_words_{period}'] = words
        results[f'word_sentiments_{period}'] = sentiments

        # Add summaries to results
        summary = summary_manager.get_summary(location, period)
        if summary:
            results[f'summary_{period}'] = summary

    return jsonify(results)

class SentimentAnalyzer:
    def __init__(self):
        # Setup stopwords
        self.stopwords = set(STOPWORDS).union(set(stopwords.words('english'))).union({
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
        })

        self.critical_terms = {
            'shooting': -0.8, 'scam': -0.7, 'unsafe': -0.7,
            'roach': -0.6, 'broke': -0.5, 'poop': -0.5,
            'trash': -0.4, 'homeless': -0.4, 'broken': -0.4,
            'rude': -0.3
        }

        self.word_contexts = defaultdict(list)
        self.cache_id = None

    def clean_text(self, text):
        """Clean and normalize text."""
        text = str(text).lower()
        text = re.sub(r"'t", " not", text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())

    def get_sentiment(self, text, rating=None):
        """Calculate sentiment score for text."""
        text = self.clean_text(text)
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity

        # Adjust for critical terms
        for term, impact in self.critical_terms.items():
            if term in text:
                polarity += impact

        # Factor in rating
        if rating is not None:
            rating_polarity = (rating - 3) / 2
            polarity = (polarity + rating_polarity * 2) / 3

        # Strengthen negative sentiments
        if polarity < 0:
            polarity *= 1.5

        # Scale to final score
        score = (polarity + 1) * 40 + 20
        return min(max(score, 20), 100), polarity

    def store_word_context(self, word, comment, sentiment, rating, date):
        """Store context for word analysis."""
        self.word_contexts[word].append({
            'comment': comment,
            'sentiment': sentiment,
            'rating': rating,
            'date': date.strftime('%Y-%m-%d')
        })

    def get_word_contexts(self, word):
        """Get stored contexts for a word."""
        return self.word_contexts.get(word, [])

    def reset_contexts(self):
        """Reset stored contexts."""
        self.word_contexts.clear()
        self.cache_id = None

    def generate_wordcloud(self, text, word_sentiments, timestamp):
        """Generate and save word cloud."""
        def color_func(word, **kwargs):
            sentiment = word_sentiments.get(word.lower(), 0)
            if sentiment < 0:
                return f"rgb({min(255, 150 + int(abs(sentiment) * 255))}, 0, 0)"
            return f"rgb(0, {min(255, 100 + int(sentiment * 255))}, 0)"

        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            stopwords=self.stopwords,
            color_func=color_func,
            max_words=100,
            collocations=False
        ).generate(text)

        filename = f"wordcloud_{timestamp}.png"
        wc.to_file(os.path.join(STATIC_DIR, filename))
        return filename


class ReviewAnalyzer:
    def __init__(self, df):
        self.df = df
        self.sentiment_analyzer = SentimentAnalyzer()

    def analyze_period(self, days_back, filtered_df=None, timestamp=None):
        """Analyze reviews for a specific time period."""
        df = filtered_df if filtered_df is not None else self.df
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_back)
        period_df = df[df["created_at"] >= cutoff_date]

        if period_df.empty:
            return 0, f"wordcloud_empty_{days_back}.png", [], {}

        # Generate cache ID
        cache_id = hashlib.md5(f"{days_back}-{df.shape[0]}-{cutoff_date}".encode()).hexdigest()
        if self.sentiment_analyzer.cache_id != cache_id:
            self.sentiment_analyzer.reset_contexts()
            self.sentiment_analyzer.cache_id = cache_id

        # Process reviews
        sentiments = []
        word_counts = Counter()
        word_sentiments = defaultdict(lambda: {"contexts": [], "polarities": []})
        collected_words = []

        for _, row in period_df.iterrows():
            score, polarity = self.sentiment_analyzer.get_sentiment(row['comment'], row.get('rating'))
            sentiments.append(score)

            words = self.sentiment_analyzer.clean_text(row['comment']).split()
            filtered_words = [w for w in words if w not in self.sentiment_analyzer.stopwords and len(w) > 2]
            collected_words.extend(filtered_words)
            word_counts.update(filtered_words)

            for word in set(filtered_words):
                word_sentiments[word]["polarities"].append(polarity)
                word_sentiments[word]["contexts"].append(row['comment'])

                self.sentiment_analyzer.store_word_context(
                    word, row['comment'], polarity,
                    row.get('rating', 3), row['created_at']
                )

        # Calculate final sentiments
        final_sentiments = {}
        for word, data in word_sentiments.items():
            avg_sentiment = sum(data["polarities"]) / len(data["polarities"])
            pos_count = sum(1 for p in data["polarities"] if p > 0)
            neg_count = sum(1 for p in data["polarities"] if p < 0)

            if pos_count > 2 * neg_count:
                avg_sentiment = min(1.0, avg_sentiment * 1.5)
            elif neg_count > 2 * pos_count:
                avg_sentiment = max(-1.0, avg_sentiment * 1.5)

            final_sentiments[word] = avg_sentiment

        # Generate word cloud
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        wc_filename = self.sentiment_analyzer.generate_wordcloud(
            " ".join(collected_words),
            final_sentiments,
            f"{days_back}_{timestamp}"
        )

        # Calculate final score
        if sentiments:
            mood_score = sum(sentiments) / len(sentiments)
            avg_rating = period_df["rating"].mean() if "rating" in period_df.columns else 3
            rating_score = (avg_rating / 5) * 100
            final_score = (mood_score + rating_score) / 2
        else:
            final_score = 50  # Default score for empty periods

        return final_score, wc_filename, word_counts.most_common(10), final_sentiments


def load_data():
    """Load and preprocess review data."""
    try:
        df = pd.read_csv(DATA_PATH)
        df['created_at'] = pd.to_datetime(df['created_at'])
        logger.info(f"Successfully loaded {len(df)} reviews")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=["location_name", "comment", "rating", "created_at"])


def clean_old_files():
    """Remove old word cloud images."""
    for f in glob.glob(os.path.join(STATIC_DIR, "wordcloud_*.png")):
        try:
            os.remove(f)
            logger.info(f"Removed old file: {f}")
        except Exception as e:
            logger.error(f"Error removing {f}: {e}")


# Initialize data
df_reviews = load_data()
review_analyzer = ReviewAnalyzer(df_reviews)


@app.route("/")
def index():
    """Render main dashboard."""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    locations = sorted(df_reviews["location_name"].unique())

    # Analyze different time periods
    periods = [90, 120, 180, 365]
    results = {}

    for period in periods:
        score, wc, words, sentiments = review_analyzer.analyze_period(period, timestamp=timestamp)
        # Get summary for the period
        summary = summary_manager.get_summary('', period)  # Empty string for no location filter

        results[period] = {
            'mood': score,
            'wordcloud': wc,
            'words': words,
            'sentiments': sentiments,
            'summary': summary if summary else {}  # Ensure summary is never None
        }

    return render_template(
        "index.html",
        locations=locations,
        mood_90=results[90]['mood'],
        mood_120=results[120]['mood'],
        mood_180=results[180]['mood'],
        mood_365=results[365]['mood'],
        wc_90=results[90]['wordcloud'],
        wc_120=results[120]['wordcloud'],
        wc_180=results[180]['wordcloud'],
        wc_365=results[365]['wordcloud'],
        top_words_90=results[90]['words'],
        top_words_120=results[120]['words'],
        top_words_180=results[180]['words'],
        top_words_365=results[365]['words'],
        word_sentiments_90=results[90]['sentiments'],
        word_sentiments_120=results[120]['sentiments'],
        word_sentiments_180=results[180]['sentiments'],
        word_sentiments_365=results[365]['sentiments'],
        summary_90=results[90]['summary'],
        summary_120=results[120]['summary'],
        summary_180=results[180]['summary'],
        summary_365=results[365]['summary'],
        now=timestamp
    )


@app.route("/word_analysis/<word>")
def word_analysis(word):
    """Get detailed analysis for a specific word."""
    contexts = review_analyzer.sentiment_analyzer.get_word_contexts(word)
    contexts.sort(key=lambda x: x['sentiment'])

    return jsonify({
        'word': word,
        'contexts': contexts,
        'total_appearances': len(contexts),
        'positive_contexts': sum(1 for c in contexts if c['sentiment'] > 0),
        'negative_contexts': sum(1 for c in contexts if c['sentiment'] < 0)
    })


if __name__ == "__main__":
    clean_old_files()
    app.run(debug=True, port=5001)