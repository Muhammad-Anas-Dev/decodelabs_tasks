"""
====================================================================
 PROJECT 3 - AI RECOMMENDATION LOGIC
 Tech Stack Recommender (Content-Based Filtering)
 DecodeLabs | Batch 2026
====================================================================

WHAT THIS SCRIPT DOES (in plain words):
  1. INGESTION -> Load job roles + their required skills from a real
                   Kaggle dataset (job_dataset.csv).
  2. PROCESS   -> Convert skills into TF-IDF vectors (numbers), then
                   measure "closeness" using Cosine Similarity.
  3. SCORING   -> Compare the user's skills against every job role.
  4. SORTING + FILTERING -> Rank scores, keep only the Top-N matches.

This follows the exact 4-step pipeline from the course slides:
   Ingestion -> Scoring -> Sorting -> Filtering
====================================================================
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------
# STEP 0: LOAD & PREPARE THE DATASET
# --------------------------------------------------------------
def load_job_profiles(csv_path: str) -> pd.DataFrame:
    """
    The raw dataset has one row PER experience-level variant of a job
    (e.g. ".NET Developer" appears 20 times: Fresher, Mid, Senior...).

    For a recommendation engine, each *job role* should be ONE item
    with ONE combined skill profile. So we group all rows by Title
    and merge their skills into a single de-duplicated skill set.
    """
    df = pd.read_csv(csv_path)

    def merge_skills(skill_series):
        all_skills = set()
        for entry in skill_series.dropna():
            # Skills are separated by ';' -> split into individual skills
            for skill in entry.split(";"):
                skill = skill.strip()
                if skill:
                    all_skills.add(skill)
        # Join back with ';' so each skill stays ONE token (not split into words)
        return "; ".join(sorted(all_skills))

    job_profiles = (
        df.groupby("Title")["Skills"]
        .apply(merge_skills)
        .reset_index()
        .rename(columns={"Skills": "SkillProfile"})
    )
    return job_profiles


# --------------------------------------------------------------
# STEP 1 & 2: VECTOR MAPPING + TF-IDF WEIGHTING (Ingestion + Scoring)
# --------------------------------------------------------------
def build_vectorizer(job_profiles: pd.DataFrame):
    """
    Custom tokenizer: split each document by ';' instead of by spaces.
    This keeps multi-word skills like 'ASP.NET MVC' as ONE feature,
    instead of breaking it into ['asp.net', 'mvc'] like default TF-IDF would.
    """
    def skill_tokenizer(text):
        return [skill.strip().lower() for skill in text.split(";") if skill.strip()]

    vectorizer = TfidfVectorizer(tokenizer=skill_tokenizer, lowercase=False, token_pattern=None)
    job_matrix = vectorizer.fit_transform(job_profiles["SkillProfile"])
    return vectorizer, job_matrix


# --------------------------------------------------------------
# STEP 3: SCORING - Compare user skills against every job (Cosine Similarity)
# --------------------------------------------------------------
def recommend_jobs(user_skills, vectorizer, job_matrix, job_profiles, top_n=3):
    # Turn the user's skill list into the SAME kind of TF-IDF vector
    user_doc = "; ".join(user_skills)
    user_vector = vectorizer.transform([user_doc])

    # Cosine similarity between user vector and ALL job vectors at once
    scores = cosine_similarity(user_vector, job_matrix).flatten()

    # --------------------------------------------------------------
    # STEP 4: SORTING + FILTERING - rank and keep only Top-N
    # --------------------------------------------------------------
    results = job_profiles.copy()
    results["MatchScore"] = scores
    results = results.sort_values("MatchScore", ascending=False)

    top_matches = results.head(top_n)
    return top_matches


# --------------------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------------------
def main():
    csv_path = "job_dataset.csv"

    print("Loading job roles and building skill profiles...")
    job_profiles = load_job_profiles(csv_path)
    print(f"Loaded {len(job_profiles)} unique job roles.\n")

    vectorizer, job_matrix = build_vectorizer(job_profiles)

    # ---- INGESTION: capture user's skills (min 3 required) ----
    print("Enter at least 3 of your skills, separated by commas.")
    print("Example: Python, SQL, Machine Learning")
    print("(Type 'exit' anytime to quit)\n")

    while True:
        raw_input_str = input("Your skills: ")

        if raw_input_str.strip().lower() == "exit":
            print("Goodbye!")
            break

        user_skills = [s.strip() for s in raw_input_str.split(",") if s.strip()]

        if len(user_skills) < 3:
            print("⚠ Please enter at least 3 skills for accurate matching.\n")
            continue

        top_matches = recommend_jobs(user_skills, vectorizer, job_matrix, job_profiles, top_n=3)

        print(f"\nBased on your skills {user_skills}, your Top 3 recommended job roles are:\n")
        rank = 1
        for _, row in top_matches.iterrows():
            score_pct = row["MatchScore"] * 100
            print(f"{rank}. {row['Title']}  —  Match Score: {score_pct:.1f}%")
            rank += 1
        print()  # blank line before next prompt


if __name__ == "__main__":
    main()
