## 🎤 10-Minute Project Presentation (8 Slides)

**Project:** Large-Scale Personalized Recommendation Engine  
**Duration:** 10 Minutes  
**Organization:** Nexvia Technologies

### Slide 1 – Problem Statement: Why Recommendation Systems?

- Modern platforms contain millions of movies and products.
- Searching the entire catalog for every user is computationally expensive.
- Users need relevant recommendations with low response time.
- A two-stage recommendation architecture helps separate candidate retrieval from detailed ranking.

**Key idea:** First retrieve relevant candidates, then rank them to improve recommendation quality.

### Slide 2 – Dataset and Time-Based Data Splitting

**Dataset:** MovieLens 25M

For the current experiment, 5,000 users were sampled.

| Dataset | Records |
|---|---:|
| Training | 591,049 |
| Validation | 73,881 |
| Testing | 73,882 |
| Movie Catalog | 8,782 |

**Time-based split:**
- 80% oldest interactions – Training
- 10% next interactions – Validation
- 10% newest interactions – Testing

**Why not random splitting?**

Random splitting can place future interactions in the training dataset while earlier interactions appear in testing. This creates data leakage and can produce overly optimistic evaluation results.

Chronological splitting better simulates recommending movies to users based on their past behavior.

### Slide 3 – System Architecture

The project follows a two-stage recommendation architecture.

**Stage 1: Candidate Retrieval**
- TensorFlow Recommenders two-tower model.
- User and item embeddings.
- FAISS similarity search.

**Stage 2: Candidate Ranking**
- Machine learning ranking model.
- Candidate scoring and reordering.
- Personalized recommendation output.

**Serving and UI**
- FastAPI recommendation backend.
- Streamlit interactive dashboard.
- TMDB movie poster integration.

See the detailed architecture in `docs/ARCHITECTURE.md`.

### Slide 4 – Stage 1: Two-Tower Retrieval + FAISS

**Objective:** Retrieve relevant candidate movies efficiently.

The two-tower model learns:
- User embeddings from user information and interaction data.
- Item embeddings from movie information.
- Similarity between user and item representations.

FAISS indexes item embeddings and retrieves the nearest candidate movies.

**Current experimental result:**

| Metric | Result |
|---|---:|
| Candidate Recall@200 | 18.7% |
| Indexed Items | 7,681 |
| Embedding Dimension | 32 |

Candidate Recall@200 measures how many relevant test movies are present in the top 200 retrieved candidates.

### Slide 5 – Stage 2: Candidate Ranking

**Objective:** Improve the ordering of retrieved movie candidates.

The ranking model receives candidate-related features and predicts recommendation scores.

The implemented pipeline:
1. Generate candidate movies using retrieval.
2. Prepare ranking examples using available interaction labels.
3. Train the ranking model.
4. Score and reorder candidate movies.
5. Evaluate the ranked recommendations on held-out test data.

**Important implementation detail:** The current project trains its ranker using the available training and validation-derived candidate examples. The test dataset is reserved for final evaluation.

**Future improvement:** Implement a more complete Deep & Cross Network (DCN) with richer user, item, genre, and interaction features.

### Slide 6 – Experimental Results

Evaluation on the sampled MovieLens test dataset:

| Recommendation Method | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| Popularity | 0.0268 | 0.0687 | 0.0415 | 0.0682 |
| Retrieval | 0.0143 | 0.0409 | 0.0224 | 0.0402 |
| Retrieval + Ranker | 0.0152 | 0.0442 | 0.0232 | 0.0436 |

**Key observations:**
- The ranker slightly improved retrieval performance.
- The popularity baseline performed better than both personalized approaches in this experiment.
- Candidate Recall@200 was 18.7%.
- More improvements are needed in retrieval coverage and ranking quality.

**Offline A/B Replay Simulation**

| Metric | Retrieval (A) | Ranker (B) |
|---|---:|---:|
| Impressions | 520 | 540 |
| Clicks | 25 | 20 |
| CTR@10 | 4.81% | 3.70% |

- Simulated p-value: 0.373
- No statistically significant difference was observed.
- These are offline replay simulation results, not live user experiments.

### Slide 7 – Challenges and Lessons Learned

**1. Cold-Start Problem**
- New users have little or no interaction history.
- Popularity-based recommendations provide a fallback.

**2. Popularity Bias**
- Popular movies can dominate recommendation results.
- Better personalization and diversity are needed.

**3. Windows and TensorFlow Compatibility**
- TensorFlow Recommenders required legacy Keras compatibility.
- Environment and dependency configuration required troubleshooting.

**4. Large Dataset Processing**
- Processing MovieLens 25M required chunk-based loading and user sampling.
- Memory usage and processing time needed attention.

**5. Model Performance**
- The ranking model did not outperform the popularity baseline.
- Candidate coverage and feature quality remain improvement areas.

**Main lesson:** A successful recommendation system requires more than model training. Data quality, candidate coverage, ranking features, and evaluation are equally important.

### Slide 8 – Future Improvements and Conclusion

**Next Steps**

1. Improve retrieval candidate coverage.
2. Add popularity correction and logQ sampling correction.
3. Introduce sequential user behavior features.
4. Improve ranking with a fully implemented DCN architecture.
5. Add Redis caching