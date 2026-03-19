import streamlit as st
import pickle
import pandas as pd
from streamlit_option_menu import option_menu
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)


# 1. Cache the heavy Machine Learning models and DataFrames
@st.cache_resource(show_spinner=False)
def load_data():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies_df = pd.DataFrame(movies_dict)
    sim = pickle.load(open('similarity.pkl', 'rb'))
    return movies_df, sim


movies, similarity = load_data()

API_KEY = "16f15271e8a769fce3b752f556e90f29"
sia = SentimentIntensityAnalyzer()


# 2. Cache API calls to prevent the UI from freezing
@st.cache_data(ttl=86400, show_spinner=False)  # Caches data for 24 hours
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        data = requests.get(url, timeout=5).json()
        if "poster_path" in data and data["poster_path"]:
            return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        return requests.get(url, timeout=5).json()
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}&language=en-US"
    try:
        data = requests.get(url, timeout=5).json()
        for video in data.get("results", []):
            if video["type"] == "Trailer":
                return f"https://www.youtube.com/watch?v={video['key']}"
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_watch_providers(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={API_KEY}"
    try:
        data = requests.get(url, timeout=5).json()

        region_data = data.get('results', {}).get('IN', {})

        # Free Streaming
        free_raw = region_data.get('free', []) + region_data.get('ads', [])
        free_streaming = []
        seen_free = set()
        for provider in free_raw:
            if provider['provider_name'] not in seen_free and provider.get('logo_path'):
                free_streaming.append({
                    "name": provider['provider_name'],
                    "logo": "https://image.tmdb.org/t/p/w45" + provider['logo_path']
                })
                seen_free.add(provider['provider_name'])

        # Paid Streaming
        paid_streaming = []
        for provider in region_data.get('flatrate', []):
            if provider.get('logo_path'):
                paid_streaming.append({
                    "name": provider['provider_name'],
                    "logo": "https://image.tmdb.org/t/p/w45" + provider['logo_path']
                })

        # Buy or Rent
        buy_rent_raw = region_data.get('buy', []) + region_data.get('rent', [])
        buy_rent = []
        seen_buy_rent = set()
        for provider in buy_rent_raw:
            if provider['provider_name'] not in seen_buy_rent and provider.get('logo_path'):
                buy_rent.append({
                    "name": provider['provider_name'],
                    "logo": "https://image.tmdb.org/t/p/w45" + provider['logo_path']
                })
                seen_buy_rent.add(provider['provider_name'])

        link = region_data.get('link', '')

        return {
            "free_streaming": free_streaming,
            "paid_streaming": paid_streaming,
            "buy_rent": buy_rent,
            "link": link
        }
    except Exception:
        return {"free_streaming": [], "paid_streaming": [], "buy_rent": [], "link": ""}


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


@st.cache_data(ttl=3600, show_spinner=False)
def trending_movies():
    url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={API_KEY}"
    try:
        return requests.get(url, timeout=5).json().get("results", [])
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def top_rated_movies():
    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}"
    try:
        return requests.get(url, timeout=5).json().get("results", [])
    except Exception:
        return []


# Helper function to display provider logos
def display_providers(movie_id):
    providers = fetch_watch_providers(movie_id)
    jw_link = providers.get("link", "#")

    if providers["free_streaming"]:
        st.write("🎁 **Free Streaming (with/without Ads):**")
        html_logos = "".join([
                                 f'<a href="{jw_link}" target="_blank"><img src="{p["logo"]}" title="{p["name"]} - Click to view on JustWatch" width="40" style="border-radius: 8px; margin-right: 8px; margin-bottom: 10px; box-shadow: 0px 2px 4px rgba(0,0,0,0.3);"></a>'
                                 for p in providers["free_streaming"]])
        st.markdown(html_logos, unsafe_allow_html=True)

    if providers["paid_streaming"]:
        st.write("📺 **Paid Subscription:**")
        html_logos = "".join([
                                 f'<a href="{jw_link}" target="_blank"><img src="{p["logo"]}" title="{p["name"]} - Click to view on JustWatch" width="40" style="border-radius: 8px; margin-right: 8px; margin-bottom: 10px; box-shadow: 0px 2px 4px rgba(0,0,0,0.3);"></a>'
                                 for p in providers["paid_streaming"]])
        st.markdown(html_logos, unsafe_allow_html=True)

    if providers["buy_rent"]:
        st.write("🛒 **Buy / Rent:**")
        html_logos = "".join([
                                 f'<a href="{jw_link}" target="_blank"><img src="{p["logo"]}" title="{p["name"]} - Click to view on JustWatch" width="40" style="border-radius: 8px; margin-right: 8px; margin-bottom: 10px; box-shadow: 0px 2px 4px rgba(0,0,0,0.3);"></a>'
                                 for p in providers["buy_rent"]])
        st.markdown(html_logos, unsafe_allow_html=True)


# --- STREAMLIT UI ---
st.title("🎬 Movie Recommender System")

with st.sidebar:
    st.markdown("## Navigation")
    menu = option_menu(
        menu_title=None,
        options=["Home", "Trending Movies", "Top Rated Movies", "Genre Filter"],
        icons=["house", "fire", "star", "funnel"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#333"},
            "nav-link-selected": {"background-color": "#ff4b4b", "color": "white", "font-weight": "normal"},
        }
    )

# HOME
if menu == "Home":
    selected_movie_name = st.selectbox(
        'Search & Select a Movie',
        movies['title'].values
    )

    if st.button('Recommend'):
        names, posters, ids = recommend(selected_movie_name)
        movie_index = movies[movies['title'] == selected_movie_name].index[0]
        movie_id = movies.iloc[movie_index].movie_id

        st.header("Selected Movie")
        col1, col2 = st.columns([1, 2])

        with col1:
            poster = fetch_poster(movie_id)
            if poster:
                st.image(poster)

        with col2:
            details = fetch_details(movie_id)
            st.write("⭐ Rating:", details.get("vote_average", "Not Available"))
            st.write("📅 Release Date:", details.get("release_date", "Not Available"))
            st.write("📝 Overview:", details.get("overview", "Not Available"))

            display_providers(movie_id)

        st.header("Recommended Movies")
        for i in range(5):
            st.markdown("---")
            col1, col2 = st.columns([1, 2])

            with col1:
                if posters[i]:
                    st.image(posters[i])

            with col2:
                st.subheader(names[i])
                details = fetch_details(ids[i])

                st.write("⭐ Rating:", details.get("vote_average", "Not Available"))
                st.write("📅 Release Date:", details.get("release_date", "Not Available"))
                st.write("📝 Overview:", details.get("overview", "Not Available"))

                display_providers(ids[i])

                trailer = fetch_trailer(ids[i])
                if trailer:
                    st.video(trailer)

# TRENDING MOVIES
elif menu == "Trending Movies":
    st.header("🔥 Trending Movies")
    trending = trending_movies()

    for movie in trending[:10]:
        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            if movie.get("poster_path"):
                poster = "https://image.tmdb.org/t/p/w500" + movie["poster_path"]
                st.image(poster)

        with col2:
            st.subheader(movie.get("title", ""))
            st.write("⭐ Rating:", movie.get("vote_average", "Not Available"))
            st.write("📅 Release Date:", movie.get("release_date", "Not Available"))
            st.write("📝 Overview:", movie.get("overview", "Not Available"))

            display_providers(movie["id"])

            trailer = fetch_trailer(movie["id"])
            if trailer:
                st.video(trailer)

# TOP RATED MOVIES
elif menu == "Top Rated Movies":
    st.header("⭐ Top Rated Movies")
    top_movies = top_rated_movies()

    for movie in top_movies[:10]:
        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            if movie.get("poster_path"):
                poster = "https://image.tmdb.org/t/p/w500" + movie["poster_path"]
                st.image(poster)

        with col2:
            st.subheader(movie.get("title", ""))
            st.write("⭐ Rating:", movie.get("vote_average", "Not Available"))
            st.write("📅 Release Date:", movie.get("release_date", "Not Available"))
            st.write("📝 Overview:", movie.get("overview", "Not Available"))

            display_providers(movie["id"])

            trailer = fetch_trailer(movie["id"])
            if trailer:
                st.video(trailer)

# GENRE FILTER
elif menu == "Genre Filter":
    st.header("🎭 Filter Movies by Genre")

    genre_options = [
        "Action", "Adventure", "Animation", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "History",
        "Horror", "Music", "Mystery", "Romance", "Science Fiction",
        "Thriller", "TV Movie", "War", "Western"
    ]

    genre = st.selectbox("Select Genre", genre_options)

    if genre:
        filtered_movies = movies[movies["tags"].str.contains(genre.lower(), na=False)]
        st.write(f"### Top 10 {genre} Movies")
        for movie in filtered_movies["title"].head(10):
            st.markdown(f"- {movie}")
