# 🎬 Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Content%20Based-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![NLTK](https://img.shields.io/badge/NLP-NLTK%20VADER-success)

A Machine Learning-based web application that recommends movies based on user interests. This project uses **Content-Based Filtering** combined with external APIs to suggest the most similar movies, direct sequels, and global streaming availability, complete with official HD posters, trailers, and plot sentiment analysis!

### 🔴 Live Demo
**Check out the live app here:** [Movie Recommendation System](https://movie-recommendation-system-superb.streamlit.app/)

**View the Model Training Code:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U6ir0mLUhg9mpAHM5_L-fTw1H5o0InNP?usp=drive_link)

---

## ✨ Key Features
* **Multi-Tab Interface:** A rich, easy-to-navigate UI featuring **Home**, **Trending Movies**, **Top Rated Movies**, **Genre Filter**, and a side-by-side **Compare Mode**.
* **Smart & Franchise Recommendations:** Get top movie suggestions based on Cosine Similarity, plus an intelligent engine that detects and prioritizes direct sequels and franchise movies automatically!
* **Global Watch Providers:** Discover exactly where to stream, rent, or buy movies free or paid across multiple countries (Powered by TMDB).
* **API Fallback Search:** If a searched movie isn't in the local dataset, the app automatically fetches its details and recommendations from the TMDB and OMDB APIs.
* **Dynamic Posters & Trailers:** Automatically fetches high-quality official posters and embeds HD YouTube trailers directly within the web app.
* **Sentiment Analysis:** Integrated NLTK's VADER to analyze the plot overview and display the thematic 'mood' of the movie (Positive, Neutral, Dark).
* **Custom Glassmorphic UI:** Features a stunning radial-gradient background, customized Streamlit components, and smooth card-based hover animations.
* **Highly Optimized:** Fast loading and efficient memory usage (model compressed to just 44MB without losing recommendation accuracy).

---

## 🧠 How It Works
This recommendation engine uses a hybrid of **Content-Based Filtering** for local movies and API-based fetching for expanding the library boundaries.

1. **Data Preprocessing:** We combine key movie attributes like Plot Overview, Genres, Top 3 Cast members, and the Director into a single "tags" paragraph.
2. **Text Vectorization:** Using the **Bag of Words** technique (CountVectorizer), we convert textual tags into vectors (5000 dimensions) after removing stop words and applying stemming.
3. **Similarity & Sequels:** We use **Cosine Similarity** to measure the angular distance. Furthermore, algorithmic checks run against the TMDB Collections API and title strings to prioritize immediate sequels!
4. **Recommendation Engine:** It displays local ML similarities, appends any franchise/sequel elements, and fetches all watch providers and YouTube trailer links dynamically.

---

## 🛠️ Tech Stack
* **Programming Language:** Python
* **Data Manipulation & ML:** Pandas, NumPy, Scikit-Learn (CountVectorizer, Cosine Similarity)
* **Natural Language Processing:** NLTK (PorterStemmer, VADER Lexicon)
* **Web Framework & UI:** Streamlit, streamlit-option-menu, Custom CSS
* **External APIs:** TMDB API (Posters, Trailers, Watch Providers, Collections, Recommendations), OMDB API

---

## 🚀 Running the Project Locally
If you want to run this project on your own machine, follow these steps:

**1. Clone the repository**
`ash
git clone https://github.com/Jaysudani05/movie-recommendation-system.git
cd movie-recommendation-system
`

**2. Install the required libraries**
`ash
pip install -r requirements.txt
`

**3. Run the Streamlit App**
`ash
streamlit run app.py
`
*(Ensure movie_dict.pkl and similarity.pkl are present in your root directory!)*

## 📂 Dataset Used
* [TMDB 5000 Movies & Credits Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## 💡 Note on Optimization
To keep the machine learning model lightweight and bypass GitHub's 100MB file limit without needing Git LFS, the 4806x4806 similarity matrix was compressed down to 44MB by casting the data type to float16 (stype(np.float16)). This preserves the recommendation accuracy while drastically reducing the deployment overhead!

---
*Developed by Jay Sudani*
