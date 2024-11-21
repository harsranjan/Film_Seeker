# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import process  # Fuzzy string matching
from nltk.sentiment import SentimentIntensityAnalyzer  # For sentiment analysis
import nltk
import requests  # To make API requests to TMDB

# Set the page configuration
st.set_page_config(page_title="Film Seeker", layout="wide")  # Set custom title and wide layout

# Ensure you have the necessary NLTK resources
nltk.download('vader_lexicon')

# Load your dataset
df = pd.read_csv('labeled_dataset.csv')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Preprocess data
df['genre'] = df['genre'].str.lower()
df['movie_name_lower'] = df['movie_name'].str.lower()

# Step 1: Encode genres using One-Hot Encoding
df_genres = df['genre'].str.get_dummies(sep='|')
df = pd.concat([df, df_genres], axis=1)

# Step 2: Process overviews using TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['overview'].fillna(''))

# Step 3: Combine all features and convert to csr_matrix
from scipy.sparse import hstack, csr_matrix

X_features = hstack([tfidf_matrix, csr_matrix(df[df_genres.columns].values)])
X_features = X_features.tocsr()  # Convert to csr_matrix for efficient indexing

# Step 4: Fit the KNN model
from sklearn.neighbors import NearestNeighbors

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(X_features)

# Create a DataFrame to map movie indices to titles
df_indices = pd.Series(df.index, index=df['movie_name_lower']).drop_duplicates()

# Function to fetch movie poster from TMDB with error handling
def fetch_movie_poster(movie_name):
    api_key = '2aff748c22f2bb94c26aa81c1e8ce3fe'  # Replace with your actual TMDB API key
    base_url = 'https://api.themoviedb.org/3/search/movie'
    params = {'api_key': api_key, 'query': movie_name}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an error for bad HTTP responses
        data = response.json()

        # Check if 'results' exist in the response and it's not empty
        if 'results' in data and data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            else:
                return None  # No poster found
        else:
            return None  # No results found for the movie

    except requests.exceptions.RequestException as e:
        # Handle any HTTP or request errors
        st.error(f"Error fetching movie poster: {e}")
        return None

# Function to get movie recommendations using KNN
def get_recommendations_knn(movie_title='', mood='', preferred_genres=[]):
    if movie_title:
        movie_title_lower = movie_title.lower()
        matched_title_info = process.extractOne(
            movie_title_lower, df['movie_name_lower'])

        if matched_title_info:
            matched_title = matched_title_info[0]
            score = matched_title_info[1]
        else:
            return f"No match found for '{movie_title}'.", None

        if score < 70:
            return f"Could not find a close match for '{movie_title}'.", None

        idx = df_indices.get(matched_title)
        input_movie_features = X_features[idx]

        distances, indices = knn_model.kneighbors(
            input_movie_features, n_neighbors=11)

        # Exclude the first one as it will be the input movie itself
        recommended_indices = indices.flatten()[1:]
        recommended_movies = df.iloc[recommended_indices][[
            'movie_name', 'year', 'genre', 'overview']]

        # Return the searched movie and recommended movies
        searched_movie = df.iloc[idx][[
            'movie_name', 'year', 'genre', 'overview']]
        return searched_movie, recommended_movies
    else:
        # For recommendations based on mood and genres
        # Filter based on preferred genres
        if preferred_genres:
            df_filtered = df[df['genre'].apply(
                lambda x: any(genre in x.lower() for genre in preferred_genres))]
        else:
            df_filtered = df

        # If a mood is provided, apply sentiment analysis to find matching movies
        if mood:
            df_filtered['sentiment_score'] = df_filtered['overview'].apply(
                lambda x: sia.polarity_scores(x)['compound'])

            # Determine if mood is positive or negative and filter accordingly
            if mood.lower() in ['happy', 'joy', 'positive', 'excited']:
                df_filtered = df_filtered[df_filtered['sentiment_score'] > 0.5]
            elif mood.lower() in ['sad', 'down', 'negative', 'upset']:
                df_filtered = df_filtered[df_filtered['sentiment_score'] < -0.5]

        # Use KNN to find similar movies
        df_filtered_indices = df_filtered.index
        if len(df_filtered_indices) == 0:
            return None, pd.DataFrame(columns=['movie_name', 'year', 'genre', 'overview'])
        X_filtered = X_features[df_filtered_indices]

        # Ensure X_filtered is in csr_matrix format
        X_filtered = X_filtered.tocsr()

        # Fit KNN on filtered data
        knn_model_filtered = NearestNeighbors(metric='cosine', algorithm='brute')
        knn_model_filtered.fit(X_filtered)

        # Choose a random movie from the filtered set as a reference
        import random
        random_idx = random.choice(df_filtered_indices)
        random_movie_features = X_features[random_idx]

        distances, indices = knn_model_filtered.kneighbors(
            random_movie_features, n_neighbors=11)

        recommended_indices = df_filtered_indices[indices.flatten()[1:]]
        recommended_movies = df.iloc[recommended_indices][[
            'movie_name', 'year', 'genre', 'overview']]

        return None, recommended_movies

# Streamlit UI
st.title("🎬 Film Seeker")

st.write("Welcome to **Film Seeker**, your personal movie recommendation system. Search for movies or get recommendations based on your mood and preferred genres.")

# Add some padding and custom styles
st.markdown(
    """
    <style>
    .movie-box {
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .movie-poster {
        display: block;
        margin: 0 auto 15px;
    }
    .movie-title {
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .movie-year {
        font-size: 14px;
        color: #666;
        text-align: center;
        margin-bottom: 10px;
    }
    .movie-genre {
        font-size: 16px;
        color: #007BFF;
        text-align: center;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

user_choice = st.selectbox("Do you want to search for a movie by name or get recommendations based on mood and genre?", ('Search', 'Recommend'))

# Search functionality
if user_choice == 'Search':
    movie_title = st.text_input("Enter a movie title:")

    if movie_title:
        searched_movie, recommended_movies = get_recommendations_knn(movie_title, '', [])

        if isinstance(searched_movie, str):
            st.write(searched_movie)  # Display error if no match found
        else:
            st.write(f"**Searched Movie:**")

            # Box-like structure for the searched movie
            with st.container():
                poster_url = fetch_movie_poster(searched_movie['movie_name'])
                col1, col2 = st.columns([1, 3])

                with col1:
                    if poster_url:
                        st.image(poster_url, width=150, use_container_width='always', caption=searched_movie['movie_name'])

                with col2:
                    st.markdown(f"<div class='movie-box'><div class='movie-title'>{searched_movie['movie_name']}</div><div class='movie-year'>{searched_movie['year']}</div><div class='movie-genre'>{searched_movie['genre']}</div><div>{searched_movie['overview']}</div></div>", unsafe_allow_html=True)

            st.write(f"**Recommended Movies based on '{searched_movie['movie_name']}'**:")

            # Display recommended movies in a box-like layout
            for index, row in recommended_movies.iterrows():
                poster_url = fetch_movie_poster(row['movie_name'])
                with st.container():
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        if poster_url:
                            st.image(poster_url, width=150, use_container_width='always')

                    with col2:
                        st.markdown(f"<div class='movie-box'><div class='movie-title'>{row['movie_name']}</div><div class='movie-year'>{row['year']}</div><div class='movie-genre'>{row['genre']}</div><div>{row['overview']}</div></div>", unsafe_allow_html=True)

# Recommendation based on mood and genres
elif user_choice == 'Recommend':
    mood = st.text_input("How are you feeling today?")
    preferred_genres = st.text_input("Enter your preferred genres (comma-separated):").split(',')
    preferred_genres = [genre.strip().lower() for genre in preferred_genres if genre.strip()]

    if mood or preferred_genres:
        _, recommended_movies = get_recommendations_knn('', mood, preferred_genres)

        if recommended_movies.empty:
            st.write("No recommendations found based on your mood and genre preferences.")
        else:
            st.write(f"Top recommendations based on your mood '{mood}' and genre preferences:")

            # Display recommended movies in a box-like layout
            for index, row in recommended_movies.iterrows():
                poster_url = fetch_movie_poster(row['movie_name'])
                with st.container():
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        if poster_url:
                            st.image(poster_url, width=150, use_container_width='always')

                    with col2:
                        st.markdown(f"<div class='movie-box'><div class='movie-title'>{row['movie_name']}</div><div class='movie-year'>{row['year']}</div><div class='movie-genre'>{row['genre']}</div><div>{row['overview']}</div></div>", unsafe_allow_html=True)
