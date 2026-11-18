
"""CineMatch AI — Movie Recommendation Engine
Run from project root:
    streamlit run app/demo.py
"""

import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data" / "processed"
POSTER_FILE = DATA / "poster_cache.json"

sys.path.insert(0, str(SRC))

from core import Recommender


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="CineMatch AI | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 80% 0%, #24121c 0%, transparent 30%),
            linear-gradient(135deg, #08090d 0%, #11131a 55%, #090a0e 100%);
        color: #f5f5f5;
    }

    [data-testid="stSidebar"] {
        background: #101116;
        border-right: 1px solid #272832;
    }

    [data-testid="stSidebar"] * {
        color: #eeeeef;
    }

    .brand {
        font-size: 27px;
        font-weight: 900;
        letter-spacing: -1.2px;
        color: white;
    }

    .brand span {
        color: #ff3158;
    }

    .hero {
        padding: 38px 34px;
        border: 1px solid #34303a;
        border-radius: 22px;
        background:
            linear-gradient(110deg, rgba(40,20,31,.96), rgba(17,18,27,.92)),
            #171821;
        margin-bottom: 28px;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255,49,88,.14);
        color: #ff6480;
        border: 1px solid rgba(255,49,88,.35);
        border-radius: 30px;
        padding: 7px 13px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.3px;
    }

    .hero h1 {
        font-size: clamp(32px, 5vw, 53px);
        line-height: 1.1;
        font-weight: 900;
        color: #ffffff;
        margin: 20px 0 12px;
    }

    .hero h1 span {
        color: #ff3158;
    }

    .hero p {
        color: #b9bac4;
        font-size: 15px;
        max-width: 650px;
        line-height: 1.7;
    }

    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: white;
        margin: 20px 0 5px;
    }

    .section-subtitle {
        color: #888b98;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .movie-title {
        color: #ffffff;
        font-size: 14px;
        font-weight: 700;
        margin-top: 9px;
        min-height: 38px;
    }

    .movie-meta {
        color: #9295a2;
        font-size: 11px;
        line-height: 1.5;
    }

    .poster-fallback {
        height: 260px;
        border-radius: 12px;
        background: linear-gradient(145deg, #29202b, #11131c);
        border: 1px solid #39313b;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ff5575;
        font-size: 48px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #ff3158, #d91e49);
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 700;
        transition: .2s ease;
    }

    .stButton > button:hover {
        background: #ff5271;
        color: white;
        border: none;
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background: #171820;
        border: 1px solid #30313c;
        border-radius: 13px;
        padding: 15px;
    }

    [data-testid="stMetricValue"] {
        color: #ff5575;
    }

    hr {
        border-color: #30313c;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD POSTER CACHE
# --------------------------------------------------

@st.cache_data
def load_poster_cache():
    if not POSTER_FILE.exists():
        return {}

    try:
        with open(POSTER_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


poster_cache = load_poster_cache()


def get_poster_url(movie):
    """Find poster URL using MovieLens item_id."""

    try:
        movie_id = str(int(float(movie["item_id"])))
        return poster_cache.get(movie_id)
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------
# LOAD RECOMMENDER
# --------------------------------------------------

@st.cache_resource
def load_recommender():
    return Recommender()


try:
    rec = load_recommender()
except Exception as error:
    st.error(f"Could not load recommendation engine: {error}")
    st.stop()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_movie_title(movie):
    return str(movie.get("title", "Unknown Movie"))


def get_movie_genres(movie):
    genres = movie.get("genres", "")

    if pd.isna(genres):
        return "Genre unavailable"

    return str(genres).replace("|", " • ")


def get_recommendations(user_id, k, mode, genres=None):
    """Call the project's existing Recommender API."""

    try:
        result = rec.recommend(
            user_id=user_id,
            k=k,
            mode=mode,
            genres=genres,
        )

        if isinstance(result, pd.DataFrame):
            return result

        return pd.DataFrame(result)

    except TypeError:
        # Compatibility fallback for implementations that don't
        # accept the optional genres parameter.
        result = rec.recommend(
            user_id=user_id,
            k=k,
            mode=mode,
        )

        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    except Exception as error:
        st.error(f"Recommendation error: {error}")
        return pd.DataFrame()


def get_catalog():
    """Load movie catalog from the processed items file."""

    items_file = DATA / "items.parquet"

    if not items_file.exists():
        return pd.DataFrame()

    try:
        return pd.read_parquet(items_file)
    except Exception as error:
        st.error(f"Could not load movie catalog: {error}")
        return pd.DataFrame()


def display_movie_card(movie, index):
    """Display one movie with a native Streamlit poster image."""

    title = get_movie_title(movie)
    genres = get_movie_genres(movie)
    poster_url = get_poster_url(movie)

    with st.container():
        if poster_url:
            try:
                st.image(
                    poster_url,
                    use_container_width=True,
                )
            except Exception:
                st.markdown(
                    '<div class="poster-fallback">🎬</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="poster-fallback">🎬</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="movie-title">{title}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="movie-meta">{genres}</div>',
            unsafe_allow_html=True,
        )


def display_movie_grid(movies, columns=5):
    if movies is None or movies.empty:
        st.info("No movies available.")
        return

    movies = movies.reset_index(drop=True)

    for start in range(0, len(movies), columns):
        group = movies.iloc[start:start + columns]
        cols = st.columns(columns)

        for position, (_, movie) in enumerate(group.iterrows()):
            with cols[position]:
                display_movie_card(movie, start + position)


def section_heading(title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="brand">CineMatch<span> AI</span> 🎬</div>',
        unsafe_allow_html=True,
    )

    st.caption("PERSONALIZED MOVIE DISCOVERY")

    st.divider()

    st.markdown("### Recommendation Settings")

    user_ids = [
        "u95080",
        "u22253",
        "u110704",
        "u139714",
        "u159567",
        "u154125",
        "u19714",
        "u102833",
        "u17669",
        "u61690",
    ]

    user_id = st.selectbox(
        "Select User",
        user_ids,
        index=0,
    )

    mode = st.selectbox(
        "Recommendation Algorithm",
        ["ranker", "retrieval"],
        index=0,
    )

    k = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
    )

    st.divider()

    st.markdown("### Discover by Genre")

    genre_filter = st.multiselect(
        "Select genres",
        [
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Horror",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ],
    )

    st.divider()

    st.caption("POWERED BY")
    st.markdown("**TensorFlow · FAISS · TMDB**")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">✦ AI-POWERED MOVIE DISCOVERY</div>
        <h1>Discover Movies<br>That <span>Understand You.</span></h1>
        <p>
            Explore personalized movie recommendations powered by
            machine learning, collaborative filtering, and intelligent
            ranking. Find your next favourite movie.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

catalog = get_catalog()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("MOVIES", f"{len(catalog):,}")

with col2:
    st.metric("POSTERS FOUND", f"{len(poster_cache):,}")

with col3:
    st.metric("ACTIVE USER", user_id)

with col4:
    st.metric("AI MODEL", mode.upper())


st.divider()


# --------------------------------------------------
# PERSONALIZED RECOMMENDATIONS
# --------------------------------------------------

section_heading(
    "✨ Recommended For You",
    "Movies selected by your personalized recommendation engine.",
)

if st.button("Generate Recommendations", use_container_width=True):

    with st.spinner("AI is finding movies for you..."):

        recommendations = get_recommendations(
            user_id=user_id,
            k=k,
            mode=mode,
            genres=genre_filter if genre_filter else None,
        )

        if not recommendations.empty:
            st.session_state["recommendations"] = recommendations
        else:
            st.session_state["recommendations"] = pd.DataFrame()


if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = get_recommendations(
        user_id=user_id,
        k=k,
        mode=mode,
        genres=genre_filter if genre_filter else None,
    )


recommendations = st.session_state["recommendations"]

if not recommendations.empty:

    display_movie_grid(recommendations, columns=5)

else:
    st.info("Click Generate Recommendations to discover movies.")


st.divider()


# --------------------------------------------------
# SEARCH MOVIE CATALOG
# --------------------------------------------------

section_heading(
    "🔎 Explore Movie Collection",
    "Search the MovieLens movie catalog.",
)

search_query = st.text_input(
    "Search movies",
    placeholder="Search for Toy Story, Batman, Titanic...",
)

if not catalog.empty:

    filtered_catalog = catalog.copy()

    if search_query:
        filtered_catalog = filtered_catalog[
            filtered_catalog["title"]
            .astype(str)
            .str.contains(search_query, case=False, na=False)
        ]

    st.caption(f"{len(filtered_catalog):,} movies found")

    display_movie_grid(
        filtered_catalog.head(10),
        columns=5,
    )

else:
    st.warning("Movie catalog is unavailable.")


st.divider()


# --------------------------------------------------
# RECENTLY LIKED MOVIES
# --------------------------------------------------

section_heading(
    "❤️ Recently Liked Movies",
    "A quick look at movies from your recommendation history.",
)

if not catalog.empty:

    if "userId" in pd.read_parquet(DATA / "train.parquet").columns:
        train = pd.read_parquet(DATA / "train.parquet")

        user_ratings = train[
            train["userId"].astype(str) == str(user_id).replace("u", "")
        ]

        liked_ids = (
            user_ratings.sort_values("rating", ascending=False)
            .head(5)["movieId"]
            .astype(str)
            .tolist()
        )

        liked_movies = catalog[
            catalog["item_id"].astype(str).isin(liked_ids)
        ]

        display_movie_grid(liked_movies, columns=5)

    else:
        st.info("User history is not available in this dataset.")

else:
    st.info("Movie catalog is unavailable.")


st.divider()


# --------------------------------------------------
# POPULAR MOVIES
# --------------------------------------------------

section_heading(
    "🔥 Explore More Movies",
    "Discover more titles from the movie catalog.",
)

if not catalog.empty:
    display_movie_grid(
        catalog.sample(
            n=min(10, len(catalog)),
            random_state=42,
        ),
        columns=5,
    )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

section_heading(
    "🤖 About CineMatch AI",
    "A two-stage personalized recommendation architecture.",
)

a, b, c = st.columns(3)

with a:
    st.markdown("### 01 · Retrieval")
    st.write(
        "TensorFlow embeddings and FAISS identify a shortlist "
        "of candidate movies."
    )

with b:
    st.markdown("### 02 · Ranking")
    st.write(
        "A machine learning ranker scores candidate movies "
        "for personalized recommendations."
    )

with c:
    st.markdown("### 03 · Evaluation")
    st.write(
        "Recall@K, NDCG@K, and offline replay simulation "
        "help evaluate recommendation quality."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:25px 0; color:#777985;">
        <div style="font-size:20px;font-weight:900;color:#ffffff;">
            CineMatch <span style="color:#ff3158;">AI</span>
        </div>
        <p>Personalized Movie Recommendation Engine</p>
        <p style="font-size:11px;">
            Built with Python · TensorFlow · FAISS · Streamlit · TMDB
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
