import pickle
import re
import datetime as dt
import nltk
import pandas as pd
import requests
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from streamlit_option_menu import option_menu

from streamlit_searchbox import st_searchbox

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
            html, body, [class*="css"], [data-testid="stAppViewContainer"] {
                font-family: sans-serif !important;
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
            }
            [data-testid="stSidebar"][aria-expanded="true"] {
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
                margin-bottom: 1.25rem;
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
API_KEY = "89c5848c6305c57ebe62e0c8d139362b"
RECOMMENDATION_LIMIT = 7
FRANCHISE_LIMIT = 2
sia = SentimentIntensityAnalyzer()


def is_released(release_date):
    if not release_date:
        return False
    try:
        release_day = dt.date.fromisoformat(release_date)
    except ValueError:
        return False
    return release_day <= dt.date.today()


def has_poster(movie):
    return bool(movie and movie.get("poster_path"))


def release_date_value(release_date):
    if not release_date:
        return 0
    try:
        return dt.date.fromisoformat(release_date).toordinal()
    except ValueError:
        return 0


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
    try:
        urls = [
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}&language=en-US",
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}&language=en",
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}",
        ]

        videos = []
        for url in urls:
            data = requests.get(url, timeout=5).json()
            videos = data.get("results", [])
            if videos:
                break

        if not videos:
            details_url = (
                f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
                "&append_to_response=videos"
            )
            data = requests.get(details_url, timeout=5).json()
            videos = data.get("videos", {}).get("results", [])

        type_priority = {
            "Trailer": 3,
            "Teaser": 2,
            "Featurette": 1,
            "Clip": 0,
            "Behind the Scenes": 0,
        }
        youtube_candidates = [v for v in videos if v.get("site") == "YouTube" and v.get("key")]
        if youtube_candidates:
            youtube_candidates.sort(
                key=lambda v: (
                    1 if v.get("official") else 0,
                    type_priority.get(v.get("type"), -1),
                    v.get("published_at", ""),
                ),
                reverse=True,
            )
            return f"https://www.youtube.com/watch?v={youtube_candidates[0]['key']}"
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tmdb_search_movies(movie_title, limit=10):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&language=en-US&query={movie_title}"
    try:
        results = requests.get(url, timeout=5).json().get("results", [])
        filtered = [m for m in results if has_poster(m) and is_released(m.get("release_date"))]
        return filtered[:limit]
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tmdb_movie_by_title(movie_title):
    results = fetch_tmdb_search_movies(movie_title, limit=1)
    if results:
        return results[0]
    return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tmdb_recommendations(movie_id, limit=RECOMMENDATION_LIMIT):
    # Only use 'similar' to strictly guarantee genre/theme matches
    url_sim = f"https://api.themoviedb.org/3/movie/{movie_id}/similar?api_key={API_KEY}&language=en-US&page=1"
    
    similar_movies = []
    seen = set()
    
    try:
        res_sim = requests.get(url_sim, timeout=5).json().get("results", [])
        for m in res_sim:
            if m.get("id") not in seen and has_poster(m) and is_released(m.get("release_date")):
                similar_movies.append(m)
                seen.add(m.get("id"))
    except Exception:
        pass

    similar_movies.sort(key=lambda m: release_date_value(m.get("release_date")), reverse=True)
    return similar_movies[:limit]


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
                if not has_poster(movie) or not is_released(movie.get("release_date")):
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
                    {"name": provider["provider_name"], "logo": "https://image.tmdb.org/t/p/w45" + provider["logo_path"]}
                )
                seen_free.add(provider_key)

        paid_streaming = []
        seen_paid = set()
        for provider in region_data.get("flatrate", []):
            provider_key = provider.get("provider_id", provider.get("provider_name"))
            if provider_key not in seen_paid and provider.get("logo_path"):
                paid_streaming.append(
                    {"name": provider["provider_name"], "logo": "https://image.tmdb.org/t/p/w45" + provider["logo_path"]}
                )
                seen_paid.add(provider_key)

        buy_rent_raw = region_data.get("buy", []) + region_data.get("rent", [])
        buy_rent = []
        seen_buy_rent = set()
        for provider in buy_rent_raw:
            provider_key = provider.get("provider_id", provider.get("provider_name"))
            if provider_key not in seen_buy_rent and provider.get("logo_path"):
                buy_rent.append(
                    {"name": provider["provider_name"], "logo": "https://image.tmdb.org/t/p/w45" + provider["logo_path"]}
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
        return {"ALL": "All Countries", "US": "United States", "IN": "India", "GB": "United Kingdom"}


def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:]

    candidates = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        details = fetch_details(movie_id)
        if not is_released(details.get("release_date")):
            continue
        poster = fetch_poster(movie_id)
        if not poster:
            continue
        candidates.append(
            {
                "title": movies.iloc[i[0]].title,
                "poster": poster,
                "id": movie_id,
                "score": i[1],
                "release_date": details.get("release_date"),
            }
        )
        if len(candidates) >= RECOMMENDATION_LIMIT * 3:
            break

    candidates.sort(
        key=lambda m: (m["score"], release_date_value(m.get("release_date"))),
        reverse=True,
    )
    candidates = candidates[:RECOMMENDATION_LIMIT]

    recommended_movies = [m["title"] for m in candidates]
    recommended_posters = [m["poster"] for m in candidates]
    recommended_ids = [m["id"] for m in candidates]

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
                if part.get("id") != movie_id and has_poster(part) and is_released(part.get("release_date")):
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
                    if rec.get("id") != movie_id and has_poster(rec) and is_released(rec.get("release_date")):
                        franchise_movies.append(rec)
            except Exception:
                pass
            # If nothing, search using the prominent first word
            if not franchise_movies and len(base_word) > 2:
                try:
                    url2 = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&language=en-US&query={base_word}"
                    for rec in requests.get(url2, timeout=5).json().get("results", []):
                        rec_title = rec.get("title", "")
                        if (
                            rec.get("id") != movie_id
                            and base_word.lower() in rec_title.lower()
                            and has_poster(rec)
                            and is_released(rec.get("release_date"))
                        ):
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
    deduped.sort(key=lambda m: release_date_value(m.get("release_date")), reverse=True)
    return deduped[:FRANCHISE_LIMIT]


def find_dataset_title(movie_query):
    query = (movie_query or "").strip().lower()
    if not query:
        return None
    title_series = movies["title"].fillna("").str.strip().str.lower()
    
    # 1. Try Exact match
    exact_matches = movies[title_series == query]
    if not exact_matches.empty:
        return exact_matches.iloc[0]["title"]
        
    # 2. Try Partial/Contains match
    partial_matches = movies[title_series.str.contains(query, regex=False, na=False)]
    if not partial_matches.empty:
        return partial_matches.iloc[0]["title"]
        
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def trending_movies():
    url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={API_KEY}"
    try:
        results = requests.get(url, timeout=5).json().get("results", [])
        strict = [m for m in results if has_poster(m) and is_released(m.get("release_date"))]
        if strict:
            return strict
        poster_only = [m for m in results if has_poster(m)]
        return poster_only
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def top_rated_movies():
    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}"
    try:
        results = requests.get(url, timeout=5).json().get("results", [])
        return [m for m in results if has_poster(m) and is_released(m.get("release_date"))]
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
    st.markdown("<div class='provider-block'><strong>Where to watch</strong></div>", unsafe_allow_html=True)
    if not has_any_provider:
        st.info("Streaming platform is not available for this country/region.")
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


def display_movie_details(movie_id, title, poster, region, show_trailer=True, details=None):
    st.markdown("<div class='movie-row'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])

    with col1:
        render_poster_full(poster, title)
        display_providers(movie_id, region)

    with col2:
        details = details or fetch_details(movie_id)
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
            else:
                st.info("Trailer not available on YouTube.")

        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


apply_custom_theme()
st.markdown(
    """
    <div class="hero-box">
        <h2 style="margin:0; font-weight:700; letter-spacing:.2px; font-family:sans-serif;">🎬 Movie Recommender System</h2>
        <p style="margin:.3rem 0 0 0;">Discover similar movies, trending titles, and where to watch them — with a smoother, card-based experience.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Navigation")
    menu = option_menu(
        menu_title=None,
        options=["Home", "Trending Movies", "Top Rated Movies", "Genre Filter"],
        icons=["house", "fire", "star", "funnel"],
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
    if "IN" in region_codes:
        default_index = region_codes.index("IN")
    else:
        default_index = region_codes.index("ALL") if "ALL" in region_codes else 0
    region = st.selectbox(
        "Watch providers region",
        region_codes,
        index=default_index,
        format_func=lambda code: region_map.get(code, code),
    )
    max_items = st.slider("How many movies to show", min_value=5, max_value=20, value=10, step=5)


if menu == "Home":
    combined_options = get_combined_selectbox_options()

    def search_movies(searchterm: str) -> list[str]:
        if not searchterm:
            return combined_options[:100]
        term = searchterm.lower()
        dataset_titles = [m for m in combined_options if term in m.lower()][:10]
        
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&language=en-US&query={searchterm}"
        try:
            results = requests.get(url, timeout=5).json().get("results", [])
            api_titles = [r["title"] for r in results if r.get("title")][:10]
        except Exception:
            api_titles = []
            
        merged = dataset_titles
        seen = {t.lower() for t in merged}
        for t in api_titles:
            if t.lower() not in seen:
                merged.append(t)
                seen.add(t.lower())
                
        # To prevent "Add: {searchterm}" in the searchbox, ensure the term is exactly in the options
        if searchterm and searchterm.lower() not in seen:
            merged.insert(0, searchterm)
            
        return merged

    selected_movie_input = st_searchbox(
        search_movies,
        key="searchbox",
        placeholder="Search & Select a Movie",
        default_options=combined_options[:50]
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
                            if not rec_poster:
                                continue
                            display_movie_details(rec_id, rec_title, rec_poster, region, details=rec)
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
                    display_movie_details(selected_movie_id, selected_title, selected_poster, region, details=tmdb_movie)

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
                                if not rec_poster:
                                    continue
                                display_movie_details(rec_id, rec_title, rec_poster, region, details=rec)
                                shown_ids.add(rec_id)

                    tmdb_recommendations = fetch_tmdb_recommendations(selected_movie_id, limit=20)
                    filtered_recommendations = [rec_movie for rec_movie in tmdb_recommendations if rec_movie.get("id") not in shown_ids][:RECOMMENDATION_LIMIT]
                    
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
                        if not rec_poster:
                            continue
                        display_movie_details(rec_id, rec_title, rec_poster, region, details=rec_movie)
                        shown_ids.add(rec_id)
                        
                    if not filtered_recommendations and not franchise_recs:
                        st.info("No alternative recommendations found for this title.")
                else:
                    suggestion_movies = fetch_tmdb_search_movies(normalized_query, limit=RECOMMENDATION_LIMIT)
                    if not suggestion_movies:
                        suggestion_movies = trending_movies()[:RECOMMENDATION_LIMIT]

                    if suggestion_movies:
                        st.subheader("✨ Recommendations")
                        shown_ids = set()

                        for movie in suggestion_movies:
                            rec_id = movie.get("id")
                            if not rec_id or rec_id in shown_ids:
                                continue
                            st.markdown("---")
                            rec_title = movie.get("title", "Untitled")
                            rec_poster = (
                                "https://image.tmdb.org/t/p/w500" + movie["poster_path"]
                                if movie.get("poster_path")
                                else None
                            )
                            if not rec_poster:
                                continue
                            display_movie_details(rec_id, rec_title, rec_poster, region, details=movie)
                            shown_ids.add(rec_id)

                        seed_id = next(iter(shown_ids), None)
                        if seed_id:
                            api_recommendations = fetch_tmdb_recommendations(seed_id, limit=10)
                            filtered_recommendations = [
                                rec_movie for rec_movie in api_recommendations if rec_movie.get("id") not in shown_ids
                            ][:RECOMMENDATION_LIMIT]
                            if filtered_recommendations:
                                st.subheader("✨ More Recommendations")
                                for rec_movie in filtered_recommendations:
                                    st.markdown("---")
                                    rec_id = rec_movie.get("id")
                                    rec_title = rec_movie.get("title", "Untitled")
                                    rec_poster = (
                                        "https://image.tmdb.org/t/p/w500" + rec_movie["poster_path"]
                                        if rec_movie.get("poster_path")
                                        else None
                                    )
                                    if not rec_poster:
                                        continue
                                    display_movie_details(rec_id, rec_title, rec_poster, region, details=rec_movie)
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
        display_movie_details(movie["id"], movie.get("title", "Untitled"), poster, region, details=movie)


elif menu == "Top Rated Movies":
    st.header("⭐ Top Rated Movies")
    top_movies = top_rated_movies()
    for movie in top_movies[:max_items]:
        st.markdown("---")
        poster = (
            "https://image.tmdb.org/t/p/w500" + movie["poster_path"] if movie.get("poster_path") else None
        )
        display_movie_details(movie["id"], movie.get("title", "Untitled"), poster, region, details=movie)


elif menu == "Genre Filter":
    st.header("🎭 Filter Movies by Genre")
    genre_options = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", 
        "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", 
        "Romance", "Science Fiction", "Thriller", "TV Movie", "War", "Western"
    ]

    genre = st.selectbox("Select Genre", genre_options)
    if genre:
        filtered_movies = movies[movies["tags"].str.contains(genre.lower(), na=False)]
        st.subheader(f"Top {max_items} {genre} movies")
        cols = st.columns(2)
        for i, movie_name in enumerate(filtered_movies["title"].head(max_items)):
            cols[i % 2].markdown(f"- {movie_name}")


