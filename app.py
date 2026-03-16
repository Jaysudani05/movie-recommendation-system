import streamlit as st
import pickle
import pandas as pd
from streamlit_option_menu import option_menu
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
#Load data

movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))

API_KEY = "16f15271e8a769fce3b752f556e90f29"

sia = SentimentIntensityAnalyzer()

# Fetch Poster

def fetch_poster(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"

    data = requests.get(url).json()

    if "poster_path" in data and data["poster_path"]:

        return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]

    return None

# Fetch Movie Details

def fetch_details(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"

    return requests.get(url).json()

#Fetch trailers

def fetch_trailer(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}&language=en-US"

    data = requests.get(url).json()

    for video in data["results"]:

        if video["type"] == "Trailer":

            return f"https://www.youtube.com/watch?v={video['key']}"

    return None

# RECOMMENDATION FUNCTION

def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []
    recommended_ids = []

    for i in movies_list:

        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)

        recommended_posters.append(fetch_poster(movie_id))

        recommended_ids.append(movie_id)

    return recommended_movies, recommended_posters, recommended_ids

# TRENDING MOVIES

def trending_movies():

    url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={API_KEY}"

    return requests.get(url).json()["results"]


# TOP RATED MOVIES

def top_rated_movies():

    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}"

    return requests.get(url).json()["results"]


# STREAMLIT UI

st.title("🎬 Movie Recommender System")

menu = option_menu(
    menu_title=None,
    options=["Home", "Trending Movies", "Top Rated Movies", "Genre Filter"],
    icons=["house", "fire", "star", "funnel"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "white", "font-size": "18px"}, 
        "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px", "--hover-color": "#333"},
        "nav-link-selected": {"background-color": "#ff4b4b", "color": "white", "font-weight": "normal"},
    }
)

# HOME

if menu == "Home":

    selected_movie_name = st.selectbox(
        'Select Movie',
        movies['title'].values
    )

    if st.button('Recommend'):

        names, posters, ids = recommend(selected_movie_name)

        movie_index = movies[movies['title'] == selected_movie_name].index[0]

        movie_id = movies.iloc[movie_index].movie_id

        st.header("Selected Movie")

        col1, col2 = st.columns([1,2])

        with col1:

            poster = fetch_poster(movie_id)

            if poster:
                st.image(poster)

        with col2:

            details = fetch_details(movie_id)

            st.write("⭐ Rating:", details.get("vote_average", "Not Available"))

            st.write("📅 Release Date:", details.get("release_date", "Not Available"))

            st.write("📝 Overview:", details.get("overview", "Not Available"))

        # RECOMMENDED MOVIES

        st.header("Recommended Movies")

        for i in range(5):

            st.markdown("---")

            col1, col2 = st.columns([1,2])

            with col1:

                if posters[i]:

                    st.image(posters[i])

            with col2:

                st.subheader(names[i])

                details = fetch_details(ids[i])

                st.write("⭐ Rating:", details.get("vote_average", "Not Available"))
                st.write("📅 Release Date:", details.get("release_date", "Not Available"))
                st.write("📝 Overview:", details.get("overview", "Not Available"))

                trailer = fetch_trailer(ids[i])

                if trailer:

                    st.video(trailer)


# TRENDING MOVIES

elif menu == "Trending Movies":

    st.header("🔥 Trending Movies")

    trending = trending_movies()

    for movie in trending[:10]:
        st.markdown("---")
        
        col1, col2 = st.columns([1,2])
        
        with col1:
            if movie.get("poster_path"):
                poster = "https://image.tmdb.org/t/p/w500" + movie["poster_path"]
                st.image(poster)
                
        with col2:
            st.subheader(movie.get("title", ""))
            
            st.write("⭐ Rating:", movie.get("vote_average", "Not Available"))
            st.write("📅 Release Date:", movie.get("release_date", "Not Available"))
            st.write("📝 Overview:", movie.get("overview", "Not Available"))
            
            trailer = fetch_trailer(movie["id"])
            if trailer:
                st.video(trailer)


# TOP RATED MOVIES

elif menu == "Top Rated Movies":

    st.header("⭐ Top Rated Movies")

    top_movies = top_rated_movies()

    for movie in top_movies[:10]:
        st.markdown("---")
        
        col1, col2 = st.columns([1,2])
        
        with col1:
            if movie.get("poster_path"):
                poster = "https://image.tmdb.org/t/p/w500" + movie["poster_path"]
                st.image(poster)
                
        with col2:
            st.subheader(movie.get("title", ""))
            
            st.write("⭐ Rating:", movie.get("vote_average", "Not Available"))
            st.write("📅 Release Date:", movie.get("release_date", "Not Available"))
            st.write("📝 Overview:", movie.get("overview", "Not Available"))
            
            trailer = fetch_trailer(movie["id"])
            if trailer:
                st.video(trailer)


# GENRE FILTER

elif menu == "Genre Filter":

    st.header("🎭 Filter Movies by Genre")

    genre = st.text_input("Enter Genre (Action, Comedy, Drama etc.)")

    if genre:

        filtered_movies = movies[movies["tags"].str.contains(genre.lower())]

        for movie in filtered_movies["title"].head(10):

            st.write(movie)

