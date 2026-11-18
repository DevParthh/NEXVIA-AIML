# 🎬 Large-Scale Personalized Movie Recommendation Engine

**AIML-03 | Nexvia Technologies Internship Project**

A two-stage personalized movie recommendation system built using Python, TensorFlow, FAISS, Machine Learning, and Streamlit. The project uses MovieLens 25M data to generate personalized movie recommendations based on user interaction history.

---

## 🚀 Project Overview

This project implements a scalable recommendation pipeline using two main stages:

1. **Candidate Retrieval:** Finds relevant movies using user and item embeddings.
2. **Candidate Ranking:** Re-ranks retrieved movies using a machine learning model.

The system also includes a FAISS similarity-search index, offline evaluation, an A/B replay simulation, and a Streamlit web application.

## ✨ Key Features

- Personalized movie recommendations
- Two-stage retrieval and ranking architecture
- TensorFlow-based recommendation model
- FAISS similarity search
- MovieLens dataset processing
- Chronological train/validation/test split
- Movie poster integration using TMDB
- Genre-based movie discovery
- FastAPI recommendation endpoint
- Streamlit interactive dashboard
- Offline recommendation evaluation
- A/B testing replay simulation

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, TensorFlow Recommenders |
| Similarity Search | FAISS |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Dataset | MovieLens 25M |
| Movie Posters | TMDB API |
| Storage | Parquet, JSON |

## 📂 Project Structure

```text
AIML-03/
│
├── app/
│   └── demo.py
│
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── train_retrieval.py
│   ├── build_faiss.py
│   ├── train_ranker.py
│   ├── evaluate.py
│   ├── ab_test.py
│   ├── core.py
│   └── serve.py
│
├── data/
│   ├── processed/
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   ├── test.parquet
│   │   ├── items.parquet
│   │   └── poster_cache.json
│   │
│   └── synthetic_backup/
│
├── artifacts/
├── .env
├── requirements.txt
└── README.md
```

## 📊 Dataset

The project uses the **MovieLens 25M dataset**.

Dataset source:

https://grouplens.org/datasets/movielens/25m/

For the current real-data experiment, 5,000 users were sampled from the dataset.

| Dataset | Records |
|---|---:|
| Training | 591,049 |
| Validation | 73,881 |
| Testing | 73,882 |
| Movie catalog | 8,782 |

The data is split chronologically to prevent future interactions from leaking into training.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AIML-03
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure TensorFlow compatibility

For the current Windows environment:

```cmd
set TF_USE_LEGACY_KERAS=1
```

## 🔄 Data Preparation

Extract the MovieLens 25M dataset and run:

```cmd
python src\data_prep.py --ml-dir "PATH_TO_ML-25M_FOLDER" --sample-users 5000
```

This prepares the training, validation, testing, and movie catalog files.

## 🧠 Model Training

Run the following commands from the project root:

```cmd
cd src
```

### Step 1: Train retrieval model

```cmd
python train_retrieval.py --epochs 3
```

### Step 2: Build FAISS index

```cmd
python build_faiss.py
```

### Step 3: Train ranking model

```cmd
python train_ranker.py --epochs 3
```

### Step 4: Evaluate recommendations

```cmd
python evaluate.py
```

### Step 5: Run offline A/B replay simulation

```cmd
python ab_test.py
```

## 🌐 Run the FastAPI Backend

From the `src` directory:

```cmd
uvicorn serve:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Example recommendation endpoint:

```text
http://127.0.0.1:8000/recommend/u95080?k=10&mode=ranker
```

## 🎨 Run the Streamlit Application

Open a new terminal in the project root:

```cmd
set TF_USE_LEGACY_KERAS=1
streamlit run app/demo.py
```

Open the application:

http://localhost:8501

The dashboard provides movie discovery, personalized recommendations, genre filters, and TMDB movie posters.

## 📈 Experimental Results

Evaluation on the sampled MovieLens test dataset:

| Recommendation Method | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| Retrieval | 0.0143 | 0.0409 | 0.0224 | 0.0402 |
| Retrieval + Ranker | 0.0152 | 0.0442 | 0.0232 | 0.0436 |
| Popularity Baseline | 0.0268 | 0.0687 | 0.0415 | 0.0682 |

**Observations:**

- The ranking stage slightly improved retrieval performance.
- The popularity baseline performed better in this experiment.
- Candidate Recall@200 was 18.7%.
- Further improvements are needed in candidate generation and ranking.

### Offline A/B Replay Simulation

| Metric | Retrieval (A) | Ranker (B) |
|---|---:|---:|
| Impressions | 520 | 540 |
| Clicks | 25 | 20 |
| CTR@10 | 4.81% | 3.70% |

The simulated comparison produced a p-value of approximately 0.373. The difference was not statistically significant.

**Note:** These are offline experimental results and simulated interactions, not results from a live production A/B test.

## 🖼️ Movie Poster Integration

The project uses the TMDB API to retrieve movie poster images.

- MovieLens movie IDs are mapped to TMDB IDs using `links.csv`.
- Poster URLs are saved in `data/processed/poster_cache.json`.
- The Streamlit application displays posters using cached URLs.

Approximately 8,714 out of 8,782 movies have poster URLs in the current cache.

TMDB:

https://www.themoviedb.org/

## 🔮 Future Improvements

- Improve retrieval candidate coverage.
- Tune the ranking model and training parameters.
- Add richer movie and user features.
- Improve cold-start recommendations.
- Add user authentication and rating history.
- Integrate online feedback and live A/B testing.
- Optimize inference latency and scalability.
- Deploy the application to a cloud platform.

## 👩‍💻 Author

**Biswabandita Mahunta**

B.Sc. Information Technology & Management

GitHub: https://github.com/biswabandita2004

LinkedIn: https://linkedin.com/in/biswabandita-mahunta-065336316

---

## 📌 Project Information

**Project:** AIML-03 – Large-Scale Personalized Recommendation Engine  
**Organization:** Nexvia Technologies  
**Category:** Machine Learning / Recommendation Systems  
**Status:** Experimental implementation completed; further model optimization is ongoing.