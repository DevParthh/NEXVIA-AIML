# Individual Reflection – Biswabandita Mahunta (AIML-03)

## 1. What I Owned

I worked on the implementation and testing of the Large-Scale Personalized Movie Recommendation Engine.

My contributions included:

- Prepared the MovieLens dataset using Python and Pandas.
- Implemented and executed chronological train, validation, and test data splitting.
- Ran the TensorFlow Recommenders two-tower retrieval model training.
- Built the FAISS index for movie embedding similarity search.
- Ran the ranking model training and investigated its evaluation performance.
- Executed Recall@K, NDCG@K, and offline A/B replay evaluation.
- Integrated TMDB movie poster URLs into the Streamlit application.
- Worked on the FastAPI recommendation service and Streamlit dashboard.
- Debugged Python environment, TensorFlow compatibility, data processing, and application issues.

I also used the generated evaluation results to identify areas where the recommendation system needs improvement.

## 2. What I Learned

Through this project, I learned:

- **Two-Stage Recommendation:** Why large recommendation systems separate candidate retrieval and candidate ranking.
- **Time-Based Data Splitting:** Why chronological splitting is important to prevent data leakage and simulate real-world recommendations.
- **Two-Tower Architecture:** How user and item embeddings represent preferences and movie characteristics.
- **FAISS:** How vector similarity search helps retrieve relevant movie candidates efficiently.
- **Recommendation Evaluation:** How Recall@K and NDCG@K measure recommendation coverage and ranking quality.
- **Cold-Start Problem:** Why new users need alternative recommendation strategies such as popularity-based recommendations.
- **Model Comparison:** Why a trained model should be compared with a simple baseline instead of assuming that a complex model is always better.
- **API Integration:** How FastAPI can expose recommendation functionality to a frontend application.

## 3. What Was Hard / What I Would Do Differently

The most challenging parts were:

- Handling a large MovieLens dataset with limited local system resources.
- Resolving TensorFlow and Keras compatibility issues in the Windows environment.
- Understanding the interaction between the retrieval model, FAISS index, and ranking model.
- Debugging data preparation and model evaluation issues.
- Integrating TMDB poster images into the Streamlit dashboard.
- Improving personalized recommendation performance because the popularity baseline performed better than the trained personalized models in my current experiment.

If I were doing this project again, I would:

- Start with a smaller dataset and establish a reliable baseline.
- Validate every pipeline stage before moving to the next stage.
- Spend more time on feature engineering and candidate coverage.
- Tune the ranking model using systematic experiments.
- Add better cold-start and recommendation-diversity strategies.
- Maintain clearer experiment logs and document results throughout development.

## 4. Tools I Used for the First Time

The following tools and technologies were used during this project:

- TensorFlow Recommenders
- FAISS
- FastAPI
- Streamlit
- TMDB API
- Parquet
- MovieLens 25M dataset

I also strengthened my practical experience with Python, Pandas, NumPy, Scikit-learn, Git, and command-line project execution.

## Final Reflection

This project helped me understand how a recommendation system works beyond theoretical machine learning concepts. I gained practical experience in data preparation, model training, vector search, ranking, evaluation, API development, and frontend integration.

One of my biggest learnings was that training a model does not automatically guarantee better recommendations. Proper evaluation, comparison with baselines, debugging, and continuous improvement are essential for building an effective machine learning system.