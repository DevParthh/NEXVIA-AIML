"""Spark job: build interaction + feature tables at MovieLens-25M scale (25M rows).
Same statistics that features.py computes in pandas for the small/sample runs.
Run:  python spark_features.py --ml-dir /path/to/ml-25m --out ../data/spark_out
Needs Java 11/17/21. On Windows use WSL2 (or Google Colab)."""
import argparse
from pyspark.sql import SparkSession, functions as F

ap = argparse.ArgumentParser()
ap.add_argument("--ml-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--csv-name", default="ratings.csv")
a = ap.parse_args()

spark = SparkSession.builder.appName("recsys-features").master("local[*]").config("spark.sql.shuffle.partitions", "64").getOrCreate()
ratings = spark.read.csv(f"{a.ml_dir}/{a.csv_name}", header=True, inferSchema=True)
movies = spark.read.csv(f"{a.ml_dir}/movies.csv", header=True, inferSchema=True)

# time-based split: cut points are timestamp quantiles
q80, q90 = ratings.approxQuantile("timestamp", [0.8, 0.9], 0.001)
ratings = ratings.withColumn("split", F.when(F.col("timestamp") <= q80, "train").when(F.col("timestamp") <= q90, "val").otherwise("test"))
train = ratings.filter("split = 'train'")

item_stats = (train.groupBy("movieId").agg(F.count("*").alias("n_ratings"), F.avg("rating").alias("avg_rating"),
              (F.sum(F.when(F.col("rating") >= 4, 1).otherwise(0)) / F.count("*")).alias("like_rate"))
              .join(movies, "movieId", "left"))
user_stats = train.groupBy("userId").agg(F.count("*").alias("n_ratings"), F.avg("rating").alias("avg_rating"),
              F.max("timestamp").alias("last_ts"))
# genre popularity per user (user history feature)
genre_hist = (train.filter("rating >= 4").join(movies, "movieId").withColumn("genre", F.explode(F.split("genres", r"\|")))
              .groupBy("userId", "genre").count())

ratings.write.mode("overwrite").partitionBy("split").parquet(f"{a.out}/interactions")
item_stats.write.mode("overwrite").parquet(f"{a.out}/item_stats")
user_stats.write.mode("overwrite").parquet(f"{a.out}/user_stats")
genre_hist.write.mode("overwrite").parquet(f"{a.out}/user_genre_hist")
print("rows:", ratings.count(), "| train items:", item_stats.count(), "| train users:", user_stats.count())
spark.stop()
