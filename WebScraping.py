import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Function to scrape multiple pages of news
def scrape_news_multiple_pages(base_url, category, class_name, pages=10):
    """
    Scrape multiple pages of news articles for a given category.
    """
    news_data = []
    for page in range(1, pages + 1):
        url = f"{base_url}/page/{page}" if "indiatoday" in base_url else base_url
        print(f"Scraping {category}, Page {page}...")
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = soup.find_all('span', {'class': class_name})
            for headline in headlines:
                title = headline.get_text().strip()  # Extract the text of the headline
                link = headline.find_parent('a')['href'] if headline.find_parent('a') else None  # Get the link
                if link:
                    full_link = f'https://edition.cnn.com{link}' if link.startswith('/') else link
                    news_data.append({'title': title, 'link': full_link, 'category': category})
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return news_data

# Define categories and their URLs (removed Science and Environment categories)
categories = {
    'Sports': {'url': 'https://edition.cnn.com/sport', 'class_name': 'container__headline-text'},
    'Tech': {'url': 'https://edition.cnn.com/business/tech', 'class_name': 'container__headline-text'},
    'Business': {'url': 'https://edition.cnn.com/business', 'class_name': 'container__headline-text'},
    'Health': {'url': 'https://edition.cnn.com/health', 'class_name': 'container__headline-text'},
    'Entertainment': {'url': 'https://edition.cnn.com/entertainment', 'class_name': 'container__headline-text'},
    'Politics': {'url': 'https://edition.cnn.com/politics', 'class_name': 'container__headline-text'},
    'World News': {'url': 'https://edition.cnn.com/world', 'class_name': 'container__headline-text'},  # New category
    'Lifestyle': {'url': 'https://edition.cnn.com/style', 'class_name': 'container__headline-text'},  # New category
}

# Collect data for all categories
all_news = []
for category, info in categories.items():
    news = scrape_news_multiple_pages(info['url'], category, info['class_name'], pages=10 if category == 'Tech' else 5)
    all_news.extend(news)

# Convert the collected data into a DataFrame
df = pd.DataFrame(all_news)
print(f"Total news articles collected: {len(df)}")
print(df.head())

# Preprocessing function
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)

# Apply preprocessing
df['processed_title'] = df['title'].apply(preprocess_text)

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3))
X_tfidf = vectorizer.fit_transform(df['processed_title'])

# Prepare labels for training
category_map = {category: idx for idx, category in enumerate(categories.keys())}
df['category_label'] = df['category'].map(category_map)

# Balance the dataset
min_samples = df['category_label'].value_counts().min()
balanced_df = df.groupby('category_label').apply(lambda x: x.sample(min_samples)).reset_index(drop=True)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    vectorizer.transform(balanced_df['processed_title']),
    balanced_df['category_label'],
    test_size=0.2,
    random_state=42
)

# Model training using Logistic Regression
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=category_map.keys(), labels=list(category_map.values())))

# Hyperparameter tuning with GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Save the model and vectorizer
joblib.dump(best_model, 'news_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Test the saved model on new data
def test_model(headlines):
    """
    Test the model with new headlines.
    """
    loaded_model = joblib.load('news_classifier_model.pkl')
    loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    processed_headlines = [preprocess_text(headline) for headline in headlines]
    X_new = loaded_vectorizer.transform(processed_headlines)
    predictions = loaded_model.predict(X_new)

    # Map predictions back to category names
    category_map_reverse = {v: k for k, v in category_map.items()}
    predicted_categories = [category_map_reverse[pred] for pred in predictions]

    for headline, category in zip(headlines, predicted_categories):
        print(f"Headline: {headline}")
        print(f"Predicted Category: {category}\n")

# Example test
test_headlines = [
    "Apple releases new iPhone with groundbreaking features",
    "India wins the cricket World Cup in a thrilling final",
    "Netflix releases its most-watched series of the year",
    "Stock market hits an all-time high as companies report profits",
    "Doctors warn about a new virus outbreak",
    "New Marvel movie breaks box office records",
    "Presidential election results bring shock to the nation",  # Political headline
    "5 healthy habits to improve your lifestyle in 2025",  # Lifestyle headline
    "World leaders meet to discuss global economic reforms"  # World News headline
]
test_model(test_headlines)
