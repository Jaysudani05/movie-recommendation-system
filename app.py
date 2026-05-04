import pickle
import re

import nltk
import pandas as pd
import requests
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from streamlit_option_menu import option_menu

nltk.download("vader_lexicon", quiet=True)

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_theme():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
            html, body, [class*="css"], [data-testid="stAppViewContainer"] {
                font-family: "Poppins", sans-serif !important;
            }
            span.material-symbols-rounded, i.material-icons {
                font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
            }
            .stApp {
                background: radial-gradient(circle at top left, #111827 0%, #0b1220 55%, #020617 100%);
                color: #e2e8f0;
            }
            [data-testid="stSidebar"] {
                background: #020617 !important;
                min-width: 280px !important;
                max-width: 280px !important;
            }
            [data-testid="stHeader"] {
                background: rgba(2, 6, 23, 0.85);
            }
            [data-testid="stToolbar"] {
                right: 1rem;
            }
            .stSelectbox div[data-baseweb="select"] > div,
            .stSlider,
            .stTextInput > div > div,
            .stMultiSelect div[data-baseweb="select"] > div {
                background-color: rgba(15, 23, 42, 0.8) !important;
                border-color: rgba(148,163,184,.3) !important;
            }
            .poster-box img {
                width: 100%;
                aspect-ratio: 2 / 3;
                object-fit: cover;
                border-radius: 12px;
                border: 1px solid rgba(148,163,184,.35);
            }
            .hero-box {
                background: linear-gradient(120deg, rgba(59,130,246,.18), rgba(15,23,42,.92));
                border: 1px solid rgba(148,163,184,.28);
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                margin-bottom: 1rem;
                box-shadow: 0 8px 20px rgba(2,6,23,.35);
            }
            .movie-card {
                border: none;
                border-radius: 14px;
                padding: 0.8rem 1rem;
                background: rgba(15, 23, 42, .56);
                transition: transform .2s ease, box-shadow .2s ease;
            }
            .movie-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 20px rgba(30,41,59,.35);
            }
            .meta-pill {
                display:inline-block;
                margin-right:8px;
                margin-top: 6px;
                margin-bottom: 6px;
                background: rgba(30,41,59,.85);
                border: 1px solid rgba(148,163,184,.32);
                border-radius: 999px;
                padding: 2px 10px;
                font-size: .84rem;
            }
            .stButton > button {
                width: 100%;
                border-radius: 10px;
                border: 1px solid rgba(96,165,250,.45);
                background: linear-gradient(90deg, #2563eb, #7c3aed);
                color: white;
                font-weight: 600;
            }
            .stSelectbox label, .stSlider label {
                font-weight: 600;
            }
            .provider-block {
                margin-top: 1rem;
                margin-bottom: 0.75rem;
            }
            .trailer-block {
                margin-top: 1rem;
            }
            .movie-row {
                margin-bottom: 1.25rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_data():
    movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
    movies_df = pd.DataFrame(movies_dict)
    sim = pickle.load(open("similarity.pkl", "rb"))
    return movies_df, sim


movies, similarity = load_data()
API_KEY = "16f15271e8a769fce3b752f556e90f29"
OMDB_API_KEY = "260f3bbb"
sia = SentimentIntensityAnalyzer()


@st.cache_data(ttl=86400, show_spinner=False)
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
        videos = data.get("results", [])
        youtube_candidates = [
            v
            for v in videos
            if v.get("site") == "YouTube" and v.get("type") in {"Trailer", "Teaser"} and v.get("key")
        ]
        if youtube_candidates:
            youtube_candidates.sort(
                key=lambda v: (
                    1 if v.get("official") else 0,
                    1 if v.get("type") == "Trailer" else 0,
                    v.get("published_at", ""),
                ),
                reverse=True,
            )
            return f"https://www.youtube.com/watch?v={youtube_candidates[0]['key']}"
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_external_movie_by_title(movie_title):
    url = "https://www.omdbapi.com/"
    try:
        data = requests.get(url, params={"apikey": OMDB_API_KEY, "t": movie_title}, timeout=5).json()
        if data.get("Response") == "True":
            return data
    except Exception:
        pass
    return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tmdb_movie_by_title(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&language=en-US&query={movie_title}"
    try:
        results = requests.get(url, timeout=5).json().get("results", [])
        if results:
            return results[0]
    except Exception:
        pass
    return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tmdb_recommendations(movie_id, limit=5):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={API_KEY}&language=en-US&page=1"
    try:
        results = requests.get(url, timeout=5).json().get("results", [])
        return results[:limit]
    except Exception:
        return []


def fetch_external_movie_by_id(imdb_id):
    url = "https://www.omdbapi.com/"
    try:
        data = requests.get(url, params={"apikey": OMDB_API_KEY, "i": imdb_id}, timeout=5).json()
        if data.get("Response") == "True":
            return data
    except Exception:
        pass
    return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_external_search(query):
    url = "https://www.omdbapi.com/"
    try:
        data = requests.get(url, params={"apikey": OMDB_API_KEY, "s": query, "type": "movie"}, timeout=5).json()
        if data.get("Response") == "True":
            return data.get("Search", [])
    except Exception:
        pass
    return []


def recommend_external_movies(base_movie, limit=5):
    base_id = base_movie.get("imdbID")
    base_title = base_movie.get("Title", "")
    genre_seed = base_movie.get("Genre", "").split(",")[0].strip()
    title_seed = " ".join(base_title.split()[:2]).strip()
    query_candidates = [base_title, genre_seed, title_seed]

    recommended = []
    seen_ids = {base_id}

    for query in query_candidates:
        if not query:
            continue
        search_results = fetch_external_search(query)
        for movie in search_results:
            imdb_id = movie.get("imdbID")
            if not imdb_id or imdb_id in seen_ids:
                continue
            movie_details = fetch_external_movie_by_id(imdb_id)
            if movie_details:
                recommended.append(movie_details)
                seen_ids.add(imdb_id)
            if len(recommended) >= limit:
                return recommended
    return recommended


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_api_suggestion_titles(limit=200):
    titles = []
    seen = set()
    try:
        for page in [1, 2]:
            url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
            results = requests.get(url, timeout=5).json().get("results", [])
            for movie in results:
                title = movie.get("title")
                if not title:
                    continue
                key = title.lower()
                if key not in seen:
                    titles.append(title)
                    seen.add(key)
                if len(titles) >= limit:
                    return titles
    except Exception:
        pass
    return titles


@st.cache_data(ttl=86400, show_spinner=False)
def get_combined_selectbox_options():
    dataset_titles = movies["title"].dropna().tolist()
    api_titles = fetch_api_suggestion_titles(limit=200)
    merged = []
    seen = set()
    for title in dataset_titles + api_titles:
        key = title.lower()
        if key not in seen:
            merged.append(title)
            seen.add(key)
    return merged


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tmdb_search_titles(query, limit=20):
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return []
    url = "https://api.themoviedb.org/3/search/movie"
    try:
        results = requests.get(
            url,
            params={"api_key": API_KEY, "language": "en-US", "query": cleaned_query, "page": 1},
            timeout=5,
        ).json().get("results", [])
        titles = []
        for movie in results:
            title = movie.get("title")
            if title:
                titles.append(title)
            if len(titles) >= limit:
                break
        return titles
    except Exception:
        return []


def get_search_suggestions(query, limit=80):
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return get_combined_selectbox_options()[:limit]

    q = cleaned_query.lower()
    dataset_matches = [
        title
        for title in movies["title"].dropna().tolist()
        if q in title.lower()
    ]
    api_matches = fetch_tmdb_search_titles(cleaned_query, limit=20)

    merged = []
    seen = set()
    for title in dataset_matches + api_matches:
        key = title.lower()
        if key not in seen:
            merged.append(title)
            seen.add(key)
        if len(merged) >= limit:
            break
    return merged


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_watch_providers(movie_id, region):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={API_KEY}"
    try:
        data = requests.get(url, timeout=5).json()
        all_results = data.get("results", {})
        if region == "ALL":
            merged_region = {"free": [], "ads": [], "flatrate": [], "buy": [], "rent": []}
            first_link = ""
            for _, region_data in all_results.items():
                for key in merged_region:
                    merged_region[key].extend(region_data.get(key, []))
                if not first_link and region_data.get("link"):
                    first_link = region_data.get("link")
            region_data = merged_region
            region_data["link"] = first_link
        else:
            region_data = all_results.get(region, {})

        free_raw = region_data.get("free", []) + region_data.get("ads", [])
        free_streaming = []
        seen_free = set()
        for provider in free_raw:
            provider_key = provider.get("provider_id", provider.get("provider_name"))
            if provider_key not in seen_free and provider.get("logo_path"):
                free_streaming.append(
                    {
                        "name": provider["provider_name"],
                        "logo": "https://image.tmdb.org/t/p/w45" + provider["logo_path"],
                    }
                )
                seen_free.add(provider_key)

        paid_streaming = []
        seen_paid = set()
        for provider in region_data.get("flatrate", []):
            provider_key = provider.get("provider_id", provider.get("provider_name"))
            if provider_key not in seen_paid and provider.get("logo_path"):
                paid_streaming.append(
                    {
                        "name": provider["provider_name"],
                        "logo": "https://image.tmdb.org/t/p/w45" + provider["logo_path"],
                    }
                )
                seen_paid.add(provider_key)

        buy_rent_raw = region_data.get("buy", []) + region_data.get("rent", [])
        buy_rent = []
        seen_buy_rent = set()
        for provider in buy_rent_raw:
            provider_key = provider.get("provider_id", provider.get("provider_name"))
            if provider_key not in seen_buy_rent and provider.get("logo_path"):
                buy_rent.append(
                    {
                        "name": provider["provider_name"],
                        "logo": "https://image.tmdb.org/t/p/w45" + provider["logo_path"],
                    }
                )
                seen_buy_rent.add(provider_key)

        link = region_data.get("link", "")
        return {
            "free_streaming": free_streaming,
            "paid_streaming": paid_streaming,
            "buy_rent": buy_rent,
            "link": link,
        }
    except Exception:
        return {"free_streaming": [], "paid_streaming": [], "buy_rent": [], "link": ""}


@st.cache_data(ttl=604800, show_spinner=False)
def fetch_provider_regions():
    url = f"https://api.themoviedb.org/3/watch/providers/regions?api_key={API_KEY}&language=en-US"
    try:
        results = requests.get(url, timeout=5).json().get("results", [])
        mapped = {}
        for item in results:
            code = item.get("iso_3166_1")
            name = item.get("english_name") or code
            if code:
                mapped[code] = name
        sorted_map = dict(sorted(mapped.items(), key=lambda x: x[1]))
        return {"ALL": "All Countries"} | sorted_map
    except Exception:
        return {
            "ALL": "All Countries",
            "US": "United States",
            "IN": "India",
            "GB": "United Kingdom",
            "CA": "Canada",
            "AU": "Australia",
        }


def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []
    recommended_ids = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
        recommended_ids.append(movie_id)

    return recommended_movies, recommended_posters, recommended_ids


def normalize_title(text):
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_franchise_movies(movie_id, title):
    franchise_movies = []
    # 1. Try TMDB Collection for official sequels
    details = fetch_details(movie_id)
    collection = details.get("belongs_to_collection")
    
    if collection and "id" in collection:
        collection_id = collection["id"]
        url = f"https://api.themoviedb.org/3/collection/{collection_id}?api_key={API_KEY}&language=en-US"
        try:
            col_data = requests.get(url, timeout=5).json()
            for part in col_data.get("parts", []):
                if part.get("id") != movie_id:
                    franchise_movies.append(part)
        except Exception:
            pass

    # 2. Fallback: Search for sequels by title matching
    if not franchise_movies and title:
        clean_title = re.sub(r'[^\w\s]', '', title).strip()
        words = clean_title.split()
        if words:
            base_word = words[0]
            # Search using full title first
            try:
                url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&language=en-US&query={title}"
                for rec in requests.get(url, timeout=5).json().get("results", []):
                    if rec.get("id") != movie_id:
                        franchise_movies.append(rec)
            except Exception:
                pass
            # If nothing, search using the prominent first word
            if not franchise_movies and len(base_word) > 2:
                try:
                    url2 = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&language=en-US&query={base_word}"
                    for rec in requests.get(url2, timeout=5).json().get("results", []):
                        rec_title = rec.get("title", "")
                        if rec.get("id") != movie_id and base_word.lower() in rec_title.lower():
                            franchise_movies.append(rec)
                except Exception:
                    pass

    # Deduplicate and return top 4
    seen = set()
    deduped = []
    for m in franchise_movies:
        if m["id"] not in seen:
            deduped.append(m)
            seen.add(m["id"])
    return deduped[:4]


def find_dataset_title(movie_query):
    query = (movie_query or "").strip().lower()
    if not query:
        return None
    title_series = movies["title"].fillna("").str.strip().str.lower()
    exact_matches = movies[title_series == query]
    if not exact_matches.empty:
        return exact_matches.iloc[0]["title"]
    return None


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


def provider_group(title, items, link):
    if not items:
        return
    st.write(title)
    html_logos = "".join(
        [
            f'<a href="{link}" target="_blank"><img src="{item["logo"]}" title="{item["name"]}" width="40" style="border-radius: 8px; margin-right: 8px; margin-bottom: 8px; box-shadow: 0px 2px 4px rgba(0,0,0,0.3);"></a>'
            for item in items
        ]
    )
    st.markdown(html_logos, unsafe_allow_html=True)


def display_providers(movie_id, region):
    providers = fetch_watch_providers(movie_id, region)
    jw_link = providers.get("link", "#")

    has_any_provider = any(
        [
            providers["free_streaming"],
            providers["paid_streaming"],
            providers["buy_rent"],
        ]
    )
    if not has_any_provider:
        return False

    provider_group("🎁 **Free Streaming (with/without Ads):**", providers["free_streaming"], jw_link)
    provider_group("📺 **Paid Subscription:**", providers["paid_streaming"], jw_link)
    provider_group("🛒 **Buy / Rent:**", providers["buy_rent"], jw_link)
    return True


def render_poster_full(poster_url, title):
    if not poster_url:
        return
    st.markdown(
        f"<div class='poster-box'><img src='{poster_url}' alt='{title} poster'></div>",
        unsafe_allow_html=True,
    )


def display_movie_details(movie_id, title, poster, region, show_trailer=True):
    st.markdown("<div class='movie-row'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])

    with col1:
        render_poster_full(poster, title)
        has_providers = fetch_watch_providers(movie_id, region)
        if any([has_providers["free_streaming"], has_providers["paid_streaming"], has_providers["buy_rent"]]):
            st.markdown("<div class='provider-block'><strong>Where to watch</strong></div>", unsafe_allow_html=True)
            provider_group("🎁 **Free Streaming (with/without Ads):**", has_providers["free_streaming"], has_providers.get("link", "#"))
            provider_group("📺 **Paid Subscription:**", has_providers["paid_streaming"], has_providers.get("link", "#"))
            provider_group("🛒 **Buy / Rent:**", has_providers["buy_rent"], has_providers.get("link", "#"))

    with col2:
        details = fetch_details(movie_id)
        overview = details.get("overview")
        release = details.get("release_date")
        rating = details.get("vote_average")
        sentiment = sia.polarity_scores(overview).get("compound", 0) if overview else 0
        mood = "😊 Positive" if sentiment > 0.2 else "😐 Neutral" if sentiment >= -0.2 else "😟 Dark"

        st.markdown(f"<div class='movie-card'><h4>{title}</h4>", unsafe_allow_html=True)
        pills = []
        if rating is not None and rating != "":
            pills.append(f"<span class='meta-pill'>⭐ {rating}</span>")
        if release:
            pills.append(f"<span class='meta-pill'>📅 {release}</span>")
        if overview:
            pills.append(f"<span class='meta-pill'>{mood}</span>")
        if pills:
            st.markdown("".join(pills), unsafe_allow_html=True)
        if overview:
            st.write(overview)

        if show_trailer:
            trailer = fetch_trailer(movie_id)
            if trailer:
                st.markdown("<div class='trailer-block'></div>", unsafe_allow_html=True)
                st.video(trailer)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



apply_custom_theme()
st.markdown(
    """
    <div class="hero-box">
        <h2 style="margin:0; font-weight:700; letter-spacing:.2px; font-family:Poppins, sans-serif;">🎬 Movie Recommender System</h2>
        <p style="margin:.3rem 0 0 0;">Discover similar movies, trending titles, and where to watch them — with a smoother, card-based experience.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Navigation")
    menu = option_menu(
        menu_title=None,
        options=["Home", "Trending Movies", "Top Rated Movies", "Genre Filter", "Compare Mode"],
        icons=["house", "fire", "star", "funnel", "columns-gap"],
        menu_icon="list",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#334155",
            },
            "nav-link-selected": {
                "background-color": "#4f46e5",
                "color": "white",
                "font-weight": "600",
            },
        },
    )

    st.markdown("---")
    region_map = fetch_provider_regions()
    region_codes = list(region_map.keys())
    default_index = region_codes.index("ALL") if "ALL" in region_codes else 0
    region = st.selectbox(
        "Watch providers region",
        region_codes,
        index=default_index,
        format_func=lambda code: region_map.get(code, code),
    )
    max_items = st.slider("How many movies to show", min_value=5, max_value=20, value=10, step=5)


def format_genres(details):
    genres = details.get("genres", [])
    genre_names = [genre.get("name") for genre in genres if genre.get("name")]
    return ", ".join(genre_names) if genre_names else "Not Available"


def format_runtime(details):
    runtime = details.get("runtime")
    if not runtime:
        return "Not Available"
    hours = runtime // 60
    minutes = runtime % 60
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"

if menu == "Home":
    combined_options = get_combined_selectbox_options()
    selected_movie_input = st.selectbox(
        "Search & Select a Movie",
        options=combined_options,
        index=None,
        placeholder="Choose an option",
        accept_new_options=True,
    )

    if st.button("Search / Recommend"):
        normalized_query = (selected_movie_input or "").strip()
        if not normalized_query:
            st.warning("Please enter a movie name first.")
        else:
            dataset_title = find_dataset_title(normalized_query)
            if dataset_title:
                selected_movie_name = dataset_title
                with st.spinner("Finding your next favorites..."):
                    names, posters, ids = recommend(selected_movie_name)
                    movie_index = movies[movies["title"] == selected_movie_name].index[0]
                    selected_movie_id = movies.iloc[movie_index].movie_id
                    selected_poster = fetch_poster(selected_movie_id)
                    franchise_recs = fetch_franchise_movies(selected_movie_id, selected_movie_name)

                st.subheader("🎯 Selected Movie")
                display_movie_details(selected_movie_id, selected_movie_name, selected_poster, region)

                st.subheader("✨ Recommendations")
                shown_ids = {selected_movie_id}

                if franchise_recs:
                    st.markdown("#### 🎬 Related / Sequels")
                    for rec in franchise_recs:
                        rec_id = rec.get("id")
                        if rec_id not in shown_ids:
                            st.markdown("---")
                            rec_title = rec.get("title", "Untitled")
                            rec_poster = "https://image.tmdb.org/t/p/w500" + rec["poster_path"] if rec.get("poster_path") else None
                            display_movie_details(rec_id, rec_title, rec_poster, region)
                            shown_ids.add(rec_id)
                
                st.markdown("#### ⭐ Similar By Content")
                for idx, name in enumerate(names):
                    if ids[idx] not in shown_ids:
                        st.markdown("---")
                        display_movie_details(ids[idx], name, posters[idx], region)
                        shown_ids.add(ids[idx])
            else:
                tmdb_movie = fetch_tmdb_movie_by_title(normalized_query)
                if tmdb_movie:
                    selected_movie_id = tmdb_movie.get("id")
                    selected_title = tmdb_movie.get("title", normalized_query)
                    selected_poster = (
                        "https://image.tmdb.org/t/p/w500" + tmdb_movie["poster_path"]
                        if tmdb_movie.get("poster_path")
                        else None
                    )
                    franchise_recs = fetch_franchise_movies(selected_movie_id, selected_title)

                    st.subheader("🎯 Selected Movie")
                    display_movie_details(selected_movie_id, selected_title, selected_poster, region)

                    st.subheader("✨ Recommendations")
                    shown_ids = {selected_movie_id}

                    if franchise_recs:
                        st.markdown("#### 🎬 Related / Sequels")
                        for rec in franchise_recs:
                            rec_id = rec.get("id")
                            if rec_id not in shown_ids:
                                st.markdown("---")
                                rec_title = rec.get("title", "Untitled")
                                rec_poster = "https://image.tmdb.org/t/p/w500" + rec["poster_path"] if rec.get("poster_path") else None
                                display_movie_details(rec_id, rec_title, rec_poster, region)
                                shown_ids.add(rec_id)

                    tmdb_recommendations = fetch_tmdb_recommendations(selected_movie_id, limit=10)
                    filtered_recommendations = [rec_movie for rec_movie in tmdb_recommendations if rec_movie.get("id") not in shown_ids][:5]
                    
                    if filtered_recommendations:
                        st.markdown("#### ⭐ Similar By Content")
                    for rec_movie in filtered_recommendations:
                        st.markdown("---")
                        rec_id = rec_movie.get("id")
                        rec_title = rec_movie.get("title", "Untitled")
                        rec_poster = (
                            "https://image.tmdb.org/t/p/w500" + rec_movie["poster_path"]
                            if rec_movie.get("poster_path")
                            else None
                        )
                        display_movie_details(rec_id, rec_title, rec_poster, region)
                        shown_ids.add(rec_id)
                        
                    if not filtered_recommendations and not franchise_recs:
                        st.info("No alternative recommendations found for this title.")
                else:
                    st.error(
                        "Movie not found in the local dataset or via API. Please try another title or a closer spelling."
                    )

elif menu == "Trending Movies":
    st.header("🔥 Trending Movies")
    trending = trending_movies()
    for movie in trending[:max_items]:
        st.markdown("---")
        poster = (
            "https://image.tmdb.org/t/p/w500" + movie["poster_path"] if movie.get("poster_path") else None
        )
        display_movie_details(movie["id"], movie.get("title", "Untitled"), poster, region)

elif menu == "Top Rated Movies":
    st.header("⭐ Top Rated Movies")
    top_movies = top_rated_movies()
    for movie in top_movies[:max_items]:
        st.markdown("---")
        poster = (
            "https://image.tmdb.org/t/p/w500" + movie["poster_path"] if movie.get("poster_path") else None
        )
        display_movie_details(movie["id"], movie.get("title", "Untitled"), poster, region)

elif menu == "Genre Filter":
    st.header("🎭 Filter Movies by Genre")
    genre_options = [
        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Family",
        "Fantasy",
        "History",
        "Horror",
        "Music",
        "Mystery",
        "Romance",
        "Science Fiction",
        "Thriller",
        "TV Movie",
        "War",
        "Western",
    ]

    genre = st.selectbox("Select Genre", genre_options)
    if genre:
        filtered_movies = movies[movies["tags"].str.contains(genre.lower(), na=False)]
        st.subheader(f"Top {max_items} {genre} movies")
        cols = st.columns(2)
        for i, movie_name in enumerate(filtered_movies["title"].head(max_items)):
            cols[i % 2].markdown(f"- {movie_name}")

elif menu == "Compare Mode":
    st.header("🆚 Compare 2 Movies")
    compare_col1, compare_col2 = st.columns(2)
    with compare_col1:
        movie_a = st.selectbox("Select first movie", movies["title"].values, key="compare_movie_a")
    with compare_col2:
        movie_b = st.selectbox("Select second movie", movies["title"].values, key="compare_movie_b")

    if st.button("Compare movies"):
        if movie_a == movie_b:
            st.warning("Please select two different movies for comparison.")
        else:
            index_a = movies[movies["title"] == movie_a].index[0]
            index_b = movies[movies["title"] == movie_b].index[0]
            movie_a_id = movies.iloc[index_a].movie_id
            movie_b_id = movies.iloc[index_b].movie_id

            details_a = fetch_details(movie_a_id)
            details_b = fetch_details(movie_b_id)
            poster_a = fetch_poster(movie_a_id)
            poster_b = fetch_poster(movie_b_id)

            left, right = st.columns(2)
            for col, title, movie_id, poster, details in [
                (left, movie_a, movie_a_id, poster_a, details_a),
                (right, movie_b, movie_b_id, poster_b, details_b),
            ]:
                with col:
                    st.subheader(title)
                    render_poster_full(poster, title)
                    st.markdown(
                        "<div class='movie-card'>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<span class='meta-pill'>⭐ {details.get('vote_average', 'Not Available')}</span>"
                        f"<span class='meta-pill'>📅 {details.get('release_date', 'Not Available')}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<span class='meta-pill'>🎭 {format_genres(details)}</span>"
                        f"<span class='meta-pill'>⏱️ {format_runtime(details)}</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(details.get("overview", "Not Available"))
                    st.markdown("<div class='provider-block'><strong>Where to watch</strong></div>", unsafe_allow_html=True)
                    display_providers(movie_id, region)
                    trailer = fetch_trailer(movie_id)
                    if trailer:
                        st.markdown("<div class='trailer-block'></div>", unsafe_allow_html=True)
                        st.video(trailer)
                    st.markdown("</div>", unsafe_allow_html=True)
