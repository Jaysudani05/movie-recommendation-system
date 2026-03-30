# 🎬 Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Content%20Based-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![NLTK](https://img.shields.io/badge/NLP-NLTK%20VADER-success)

A Machine Learning-based web application that recommends movies based on user interests. This project uses a **Content-Based Filtering** approach to suggest the top 5 most similar movies to your search, complete with their official posters, detailed information, trailers, and sentiment analysis!

### 🔴 Live Demo
**Check out the live app here:** [Movie Recommendation System](https://movie-recommendation-system-superb.streamlit.app/)

**View the Model Training Code:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U6ir0mLUhg9mpAHM5_L-fTw1H5o0InNP?usp=drive_link)

---

---

## ✨ Key Features
* **Multi-Tab Interface:** A rich, easy-to-navigate UI featuring dedicated sections for **Home (Recommendations)**, **Trending Movies**, **Top Rated Movies**, and a **Genre Filter**.
* **Discover & Explore:** Users can easily browse through current trending hits, all-time highest-rated classics, or filter movies by their favorite genres.
* **Personalized Recommendations:** Get the top 5 most relevant movie suggestions based on your selection using Cosine Similarity.
* **Comprehensive Movie Details:** Instantly view essential movie information including **Ratings**, **Release Dates**, and detailed **Plot Overviews**.
* **Embedded Trailers:** Watch the official HD YouTube trailers directly within the web app without leaving the page!
* **Dynamic Posters:** Automatically fetches high-quality, official movie posters in real-time using the TMDB API.
* **Sentiment Analysis:** Integrated NLTK's VADER to analyze the sentiment of movie-related text.
* **Highly Optimized:** Fast loading and efficient memory usage (model compressed to just 44MB without losing recommendation accuracy).

---

## 🧠 How It Works
This recommendation engine relies on **Content-Based Filtering**. It doesn't rely on user ratings or history; instead, it looks at the metadata of the movies themselves.

1. **Data Preprocessing:** We combine key movie attributes like Plot Overview, Genres, Keywords, Top 3 Cast members, and the Director into a single "tags" paragraph for each movie.
2. **Text Vectorization:** Using the **Bag of Words** technique (`CountVectorizer`), we convert these textual tags into mathematical vectors (5000 dimensions) after removing English stop words and applying stemming.
3. **Calculating Similarity:** We use **Cosine Similarity** to measure the angular distance between these vectors in a high-dimensional space.
4. **Recommendation:** When a user selects a movie, the system fetches the 5 closest movie vectors and displays them along with their detailed info and multimedia fetched dynamically.

---

## 🛠️ Tech Stack
* **Programming Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (CountVectorizer, Cosine Similarity)
* **Natural Language Processing:** NLTK (PorterStemmer, VADER)
* **Web Framework:** Streamlit
* **API:** TMDB API (for fetching movie posters, details, lists, and video keys)

---

## 🚀 Running the Project Locally
If you want to run this project on your own machine, follow these steps:

**1. Clone the repository**
```
git clone [https://github.com/Jaysudani05/movie-recommendation-system.git](https://github.com/Jaysudani05/movie-recommendation-system.git)

cd movie-recommendation-system
```
**2. Install the required libraries**
```
pip install -r requirements.txt
```

**3. Run the Streamlit App**

```
streamlit run app.py
```

## 📂 Dataset Used
* [TMDB 5000 Movies & Credits Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## 💡 Note on Optimization
To keep the machine learning model lightweight and bypass GitHub's 100MB file limit without needing Git LFS, the 4806x4806 similarity matrix was compressed down to 44MB by casting the data type to float16 (astype(np.float16)). This preserves the recommendation accuracy while drastically reducing the deployment overhead!
---
*Developed by Jay Sudani*
