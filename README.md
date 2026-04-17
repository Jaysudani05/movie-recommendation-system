# 🎬 Movie Recommendation System

A production-style **Streamlit movie discovery app** powered by a content-based recommendation engine and live TMDB metadata.

It helps users:
- find similar movies instantly,
- explore trending/top-rated titles,
- check region-specific watch providers,
- and compare two movies side-by-side.

---

## Table of Contents
- [Live Links](#live-links)
- [Key Features](#key-features)
- [Product Walkthrough](#product-walkthrough)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [How Recommendations Work](#how-recommendations-work)
- [Compare Mode](#compare-mode)
- [Performance & Caching](#performance--caching)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Dataset & Credits](#dataset--credits)

---

## Live Links
- **Web App:** https://movie-recommendation-system-superb.streamlit.app/
- **Model notebook (Colab):** https://colab.research.google.com/drive/1U6ir0mLUhg9mpAHM5_L-fTw1H5o0InNP?usp=drive_link

---

## Key Features

### 1) Smart Recommendations
- Content-based filtering with precomputed cosine similarity.
- Fast top-5 recommendations for any selected movie.

### 2) Rich Movie Detail Cards
- Poster, rating, release date, overview.
- Runtime and genres (where available from TMDB).
- Embedded trailer playback.
- Sentiment mood tag (VADER).

### 3) Region-Aware Watch Providers
- Dynamic region list from TMDB provider regions endpoint.
- Grouped provider results:
  - Free / Ad-supported
  - Subscription
  - Buy / Rent
- User-facing fallback message if no providers are found.

### 4) Discovery Views
- **Home**: movie selection + recommendations.
- **Trending Movies**: weekly trending feed.
- **Top Rated Movies**: critically acclaimed picks.
- **Genre Filter**: quick genre-based browsing.
- **Compare Mode**: side-by-side comparison of two movies.

### 5) UI/UX
- Dark-themed interface with modern typography.
- Reusable card-based components.
- Configurable item count per listing.

---

## Product Walkthrough

1. Open app and choose region from **Watch providers region**.
2. Use **Home** to select a movie and get top recommendations.
3. Inspect poster, details, platforms, and trailer.
4. Use **Compare Mode** to evaluate two movies in parallel.
5. Explore **Trending**, **Top Rated**, or **Genre Filter** for discovery.

---

## Architecture

### Data Layer
- `movie_dict.pkl`: movie metadata for app lookup.
- `similarity.pkl`: precomputed similarity matrix for fast retrieval.

### Service Layer
- TMDB API wrappers (details, posters, trailers, watch providers, regions).
- Caching decorators for reduced API overhead and faster UI response.

### Presentation Layer
- Streamlit UI pages + reusable rendering helpers for consistency.

---

## Tech Stack
- **Language:** Python
- **Framework:** Streamlit
- **ML/NLP:** Scikit-learn, NLTK (VADER)
- **Data:** Pandas, Pickle
- **API:** TMDB

---

## Project Structure

```text
movie-recommendation-system/
├── app.py                 # Main Streamlit app
├── movie_dict.pkl         # Movie metadata
├── similarity.pkl         # Similarity matrix
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/Jaysudani05/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```
Then open the local URL shown in the terminal (commonly `http://localhost:8501`).

---

## Configuration

### TMDB API Key
The app currently uses a TMDB API key in code.
For production, move it to an environment variable:

```bash
export TMDB_API_KEY="your_api_key_here"
```

Then update `app.py` to read via `os.getenv("TMDB_API_KEY")`.

---

## How Recommendations Work

1. Build a movie "tags" representation from metadata.
2. Vectorize text features (Bag of Words).
3. Compute cosine similarity between movie vectors.
4. Return top nearest neighbors as recommendations.

---

## Compare Mode

Compare two selected movies side-by-side with:
- poster,
- ratings and release date,
- genres and runtime,
- watch providers by selected region,
- trailer embeds.

---

## Performance & Caching

The app uses:
- `@st.cache_resource` for heavy static loads,
- `@st.cache_data` for TMDB responses and computed data,

which reduces repeated API calls and improves responsiveness.

---

## Known Limitations

- Provider availability depends on TMDB regional data quality.
- Some movies may not have trailer/provider/runtime metadata.
- Recommendation quality depends on source metadata richness.

---

## Roadmap

- Add user watchlist and favorites.
- Add advanced filters (year, language, runtime range).
- Add explainable recommendations (“why this was recommended”).
- Add tests for API wrappers and UI helper functions.

---

## Contributing

Contributions are welcome.
1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a pull request with a clear summary

---

## Dataset & Credits

- Dataset: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- Movie metadata/media: https://www.themoviedb.org/

Built by **Jay Sudani**.
