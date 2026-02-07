from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit

def calculate_rewards():
    # 1. Start Spark
    spark = SparkSession.builder \
        .appName("StreamSafe-Reward-Construction") \
        .getOrCreate()

    print("--- SPARK STARTED ---")

    # 2. Load the Data
    # In a real system, these would be huge log files.
    df_decisions = spark.read.option("multiline", "true").json("decisions.json")
    df_overrides = spark.read.option("multiline", "true").json("overrides.json")

    print("--- DATA LOADED ---")
    df_decisions.show()
    df_overrides.show()

    # 3. Join the Data
    # We look for matches where the message_id is the same
    # "left" join means: Keep all bot decisions, even if there is no human override.
    joined_df = df_decisions.join(
        df_overrides, 
        on="message_id", 
        how="left"
    ).select(
        df_decisions.message_id,
        df_decisions.action.alias("bot_action"),
        df_decisions.risk_score,
        col("correction").alias("human_correction") # Rename for clarity
    )

    # 4. Compute Reward (The "Secret Sauce")
    # Logic:
    # - If Human UNBANS what Bot BANNED -> Reward = -1.0 (Huge Punishment)
    # - If Human BANS what Bot IGNORED -> Reward = -1.0 (Missed a troll)
    # - If Human matches Bot (or does nothing) -> Reward = +0.1 (Small "Good Job" cookie)
    
    scored_df = joined_df.withColumn(
        "reward",
        when(
            (col("bot_action") == "BAN") & (col("human_correction") == "UNBAN"), 
            -1.0
        ).otherwise(0.1)
    )

    print("--- SCORED RESULTS ---")
    scored_df.show()

    # 5. Save Training Data (Parquet)
    # Parquet is a compressed binary format used for ML training
    scored_df.write.mode("overwrite").parquet("training_data.parquet")
    print("--- SAVED TO PARQUET ---")

    spark.stop()

if __name__ == "__main__":
    calculate_rewards()