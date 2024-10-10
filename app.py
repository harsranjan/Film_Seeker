import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  # Fuzzy string matching
from nltk.sentiment import SentimentIntensityAnalyzer  # For sentiment analysis
import nltk
import requests  # To make API requests to TMDB

# Ensure you have the necessary NLTK resources
nltk.download('vader_lexicon')

# Load your dataset
df = pd.read_csv('labeled_dataset.csv')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Step 1: Combine the genre and overview into a single feature for similarity calculation
df['combined_features'] = df['genre'] + ' ' + df['overview']

# Step 2: Create a Bag of Words model using CountVectorizer
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(df['combined_features'])

# Step 3: Calculate cosine similarity between the movies
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Step 4: Create a DataFrame to map movie indices to titles (in lowercase for fuzzy matching)
df['movie_name_lower'] = df['movie_name'].str.lower()
df_indices = pd.Series(df.index, index=df['movie_name_lower']).drop_duplicates()

# Function to fetch movie poster from TMDB with error handling
def fetch_movie_poster(movie_name):
    api_key = '141fa9dc1b8fad4fb0072848a6a96e2b'  # Replace with your actual TMDB API key
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

# Function to get movie recommendations
def get_recommendations(movie_title, mood, preferred_genres, cosine_sim=cosine_sim):
    movie_title_lower = movie_title.lower()
    similar_movies = pd.DataFrame()

    if movie_title_lower:
        matched_title_info = process.extractOne(movie_title_lower, df['movie_name_lower'])

        if matched_title_info:
            matched_title = matched_title_info[0]
            score = matched_title_info[1]
        else:
            return f"No match found for '{movie_title}'."

        if score < 70:
            return f"Could not find a close match for '{movie_title}'."

        idx = df_indices.get(matched_title)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Get the top 10 recommendations

        movie_indices = [i[0] for i in sim_scores]
        similar_movies = df.iloc[movie_indices][['movie_name', 'year', 'genre', 'overview']]

        # Return the matched movie (searched movie) separately
        searched_movie = df.iloc[idx][['movie_name', 'year', 'genre', 'overview']]
        return searched_movie, similar_movies
    else:
        similar_movies = df[['movie_name', 'year', 'genre', 'overview']]
        return None, similar_movies

# Streamlit UI
st.title("Movie Recommendation System")

user_choice = st.selectbox("Do you want to search for a movie by name or get recommendations based on mood and genre?", ('Search', 'Recommend'))

if user_choice == 'Search':
    movie_title = st.text_input("Enter a movie title:")
    
    if movie_title:
        searched_movie, recommended_movies = get_recommendations(movie_title, '', [])
        
        if isinstance(searched_movie, str):
            st.write(searched_movie)  # Display error if no match found
        else:
            st.write(f"**Searched Movie:**")
            st.write(f"**{searched_movie['movie_name']} ({searched_movie['year']})** - {searched_movie['genre']}")
            st.write(f"**Description:** {searched_movie['overview']}")
            
            poster_url = fetch_movie_poster(searched_movie['movie_name'])
            if poster_url:
                st.image(poster_url, width=150)
            
            st.write(f"**Recommended Movies based on '{searched_movie['movie_name']}'**:")
            for index, row in recommended_movies.iterrows():
                st.write(f"**{row['movie_name']} ({row['year']})** - {row['genre']}")
                st.write(f"**Description:** {row['overview']}")
                poster_url = fetch_movie_poster(row['movie_name'])
                if poster_url:
                    st.image(poster_url, width=150)

elif user_choice == 'Recommend':
    mood = st.text_input("How are you feeling today?")
    preferred_genres = st.text_input("Enter your preferred genres (comma-separated):").split(',')
    preferred_genres = [genre.strip().lower() for genre in preferred_genres]

    if mood:
        _, recommended_movies = get_recommendations('', mood, preferred_genres)

        if isinstance(recommended_movies, str):
            st.write(recommended_movies)
        else:
            st.write(f"Top recommendations based on your mood '{mood}' and genre preferences:")
            for index, row in recommended_movies.iterrows():
                st.write(f"**{row['movie_name']} ({row['year']})** - {row['genre']}")
                st.write(f"**Description:** {row['overview']}")
                poster_url = fetch_movie_poster(row['movie_name'])
                if poster_url:
                    st.image(poster_url, width=150)
