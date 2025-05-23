#% The goal of this notebook is to create a recommender system for books using collaborative filtering and content-based methods. The dataset contains user interactions with books, and the task involves predicting which books a user might like based on their past interactions and the content of the books.
#% The recommender system will be evaluated using precision and recall metrics, and various methods such as TF-IDF, Google API similarity, BERT embeddings, and collaborative filtering (item-based and user-based) will be employed to generate recommendations.
#% The notebook is structured to first explore the data, create user-item matrices, and then implement different recommendation techniques. The final results will be saved in CSV files for further analysis.  

# %%
#Library
from collections import defaultdict
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split


# %% [markdown]
# # Recommender Systems
# | Recommender Type     | Similarity Between | Based On           | Example Statement                                      |
# |----------------------|--------------------|--------------------|--------------------------------------------------------|
# | CF – Item-Item       | Items              | User behavior      | “You liked A, others who liked A also liked B”         |
# | CF – User-User       | Users              | User behavior      | “People like you liked B, so you might too”            |
# | Content-Based        | Items              | Item text/content  | “These books are similar in description/topic”         |
# | Hybrid               | Items              | Content + Behavior | “You liked A; B is similar and liked by others too”    |
# 

# %% [markdown]
# ## Task 1: Exploring
# 
# 

# %% [markdown]
# #### Step 1:Get the data

# %%
# Load the datasets

interactions = pd.read_csv('https://raw.githubusercontent.com/linneverh/MachineLearning/main/interactions_train.csv')

#FOR: Google enhanced & ISBN enhanced - author_date_title_subjects priority
items1 = pd.read_csv("https://media.githubusercontent.com/media/ML-brooowss/ML/refs/heads/main/final_items/author_date_title_subjects/embeddings_part1.csv")
items2 = pd.read_csv("https://media.githubusercontent.com/media/ML-brooowss/ML/refs/heads/main/final_items/author_date_title_subjects/embeddings_part2.csv")
items = pd.concat([items1, items2])

#rename columns
interactions = interactions.rename(columns={'u': 'user_id', 'i': 'book_id', 't': 'timestamp'})
items=items.rename(columns={'i':'book_id'})

# Display the first rows of the updated interactions DataFrame
display(interactions.head())
display(items.head())

# Display the first rows of each dataset
display(interactions.head())
display(items.head())

# %% [markdown]
# 
# #### Step 2: Check the Number of interactions, users and books

# %%
n_users = interactions.user_id.nunique()
n_items = items.book_id.nunique()
print(f'Number of users = {n_users}, \n Number of books = {n_items} \n Number of interactions = {len(interactions)}')


# %% [markdown]
# 
# #### Step 3: Split the Data into Training and Test Sets

# %%
# let's first sort the interactions by user and time stamp
interactions = interactions.sort_values(["user_id", "timestamp"])
interactions.head(100)

# %%
interactions["pct_rank"] = interactions.groupby("user_id")["timestamp"].rank(pct=True, method='dense')
interactions.reset_index(inplace=True, drop=True)
interactions.head(10)

# %% [markdown]
# Now all remains to do is to pick the first 80% of the interactions of each user in the training set and the rest in the test set. We can do so using the `pct_rank` column.

# %%
train_data = interactions[interactions["pct_rank"] < 0.8]
test_data = interactions[interactions["pct_rank"] >= 0.8]

# %%
print("Training set size:", train_data.shape[0])
print("Testing set size:", test_data.shape[0])

# %% [markdown]
# ## Task 2: Creating User-Item Matrices for Implicit Feedback
# 

# %%
print('number of users =', n_users, '| number of movies =', n_items)

# %% [markdown]
# #### Step 1: Define the Function to Create the Data Matrix
# 

# %%
# Define a function to create the data matrix
def create_data_matrix(data, n_users, n_items):
    """
    This function returns a numpy matrix with shape (n_users, n_items).
    Each entry is a binary value indicating positive interaction.
    """
    data_matrix = np.zeros((n_users, n_items))
    data_matrix[data["user_id"].values, data["book_id"].values] = 1
    return data_matrix

# %% [markdown]
# #### Step 2: Create the Training and Testing Matrices
# 
# Now we can use the function to create matrices for both the training and testing data. Each cell in the matrix will show a 1 if there was a positive interaction in the training or testing data, and a 0 otherwise.

# %%
entire_data=create_data_matrix(interactions, n_users, n_items)

# %%
# Create the training and testing matrices
train_data_matrix = create_data_matrix(train_data, n_users, n_items)
test_data_matrix = create_data_matrix(test_data, n_users, n_items)

# Display the matrices to understand their structure
print('train_data_matrix')
print(train_data_matrix)
print("number of non-zero values: ", np.sum(train_data_matrix))
print('test_data_matrix')
print(test_data_matrix)
print("number of non-zero values: ", np.sum(test_data_matrix))


# %%
#give the dimensions of matrices
print("Train data matrix dimensions:", train_data_matrix.shape)
print("Test data matrix dimensions:", test_data_matrix.shape)

# %% [markdown]
# #### Basic Definitions

# %%
# Recommendation frame generation
def create_recommendation_table(user_predictions, top_n=10, separator=" "):
    """
    Creates a table of top-N recommendations for each user.

    Args:
        user_predictions (numpy.ndarray): Rows = users, columns = items. Predicted scores.
        top_n (int): Number of top recommendations per user.
        separator (str): Delimiter to join recommended book IDs.

    Returns:
        pandas.DataFrame: Columns = ['user_id', 'recommendation'].
    """
    recommendations = []
    num_users = user_predictions.shape[0]

    for user_id in range(num_users):
        top_items = np.argsort(user_predictions[user_id, :])[-top_n:][::-1]
        recommendations.append({
            'user_id': user_id,
            'recommendation': separator.join(map(str, top_items))
        })

    return pd.DataFrame(recommendations)

# %%
# Def for the precision_recall_at_k function
def precision_recall_at_k(prediction, ground_truth, k=10):
    """
    Calculates Precision@K and Recall@K for top-K recommendations.
    Parameters:
        prediction (numpy array): The predicted interaction matrix with scores.
        ground_truth (numpy array): The ground truth interaction matrix (binary).
        k (int): Number of top recommendations to consider.
    Returns:
        precision_at_k (float): The average precision@K over all users.
        recall_at_k (float): The average recall@K over all users.
    """
    num_users = prediction.shape[0]
    precision_at_k, recall_at_k = 0, 0

    for user in range(num_users):
        # TODO: Get the indices of the top-K items for the user based on predicted scores
        top_k_items = np.argsort(prediction[user, :])[-k:]

        # TODO: Calculate the number of relevant items in the top-K items for the user
        relevant_items_in_top_k = np.isin(top_k_items, np.where(ground_truth[user, :] == 1)[0]).sum()

        # TODO: Calculate the total number of relevant items for the user
        total_relevant_items = ground_truth[user, :].sum()

        # Precision@K and Recall@K for this user
        precision_at_k += relevant_items_in_top_k / k
        recall_at_k += relevant_items_in_top_k / total_relevant_items if total_relevant_items > 0 else 0

    # Average Precision@K and Recall@K over all users
    precision_at_k /= num_users
    recall_at_k /= num_users

    return precision_at_k, recall_at_k

# %%
# Create random splits def.
def random_split_per_user(interactions_df, test_size=0.2):
    train_list = []
    test_list = []
    for user_id, user_df in interactions_df.groupby('user_id'):
        train_df, test_df = train_test_split(user_df, test_size=test_size)
        train_list.append(train_df)
        test_list.append(test_df)
    return pd.concat(train_list), pd.concat(test_list)

# %%
# Define the function to predict interactions based on item similarity
def item_based_predict(interactions, similarity, epsilon=1e-9):
    """
    Predicts user-item interactions based on item-item similarity.
    Parameters:
        interactions (numpy array): The user-item interaction matrix.
        similarity (numpy array): The item-item similarity matrix.
        epsilon (float): Small constant added to the denominator to avoid division by zero.
    Returns:
        numpy array: The predicted interaction scores for each user-item pair.
    """
    # np.dot does the matrix multiplication. Here we are calculating the
    # weighted sum of interactions based on item similarity
    pred = similarity.dot(interactions.T) / (similarity.sum(axis=1)[:, np.newaxis] + epsilon)
    return pred.T  # Transpose to get users as rows and items as columns

# %% [markdown]
# ## Content-based

# %% [markdown]
# ### TF-IDF
# w. ['Publisher', 'Subjects', 'google_api_title', 'author_clean', 'ISBN']<br>
# Mean Precision@10 = 0.0149 <br>
# Mean Recall@10    = 0.091

# %%
#TF-IDF

# STEP 1: Build and clean the combined text feature
text_fields = ['Publisher', 'Subjects', 'google_api_title', 'author_clean', 'ISBN']
items['combined_text'] = items[text_fields].fillna('').agg(' '.join, axis=1)

# # STEP 2: Align items with those used in the train_data_matrix (e.g., by book_id)
# # to ensure the order of books in the TF-IDF matrix exactly matches the item columns in the collaborative filtering matrix, so similarity scores align correctly with item IDs.
items_ordered = items.set_index('book_id').loc[range(entire_data.shape[1])]

# # STEP 3: Compute TF-IDF matrix and cosine similarity
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(items_ordered['combined_text'])

# # Cosine similarity between item vectors
tfidf_sim = cosine_similarity(tfidf_matrix)

# %%
# Calculate the item-based predictions for positive interactions
item_tfidf_prediction = item_based_predict(entire_data, tfidf_sim)
print("Predicted Interaction Matrix:")
print(item_tfidf_prediction)
print(item_tfidf_prediction.shape)

# %%
# Create df
item_tfidf_recommendations_df = create_recommendation_table(item_tfidf_prediction, top_n=10, separator=" ")

# Save and display
item_tfidf_recommendations_df.to_csv('item_tfidf_recommendations.csv', index=False)

print("\nItem-based Recommendations:")
display(item_tfidf_recommendations_df)

# %%
precision_item_k, recall_item_k = precision_recall_at_k(item_tfidf_prediction, test_data_matrix, k=10)
print('Item-based EMBED Precision@K:', precision_item_k)
print('Item-based EMBED Recall@K:', recall_item_k)

# %% [markdown]
# ### Google API similarity <BR>
# Mean Precision@K: 0.04866037254401807 <BR>
# Mean Recall@K: 0.2707247031495884

# %%
# Select only the item IDs in the training data matrix
train_item_ids = range(entire_data.shape[1])

# Ensure correct item order by aligning to the item indices used in the train matrix
items_ordered = items.set_index('book_id').loc[train_item_ids]

# Parse the embedding strings into numpy arrays
items_ordered['embedding'] = items_ordered['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

# Drop rows with missing or malformed embeddings (if any)
valid_items = items_ordered[items_ordered['embedding'].notna()].reset_index(drop=True)

# Stack embeddings into a matrix
embedding_matrix = np.vstack(valid_items['embedding'].values)

# Compute cosine similarity
embedding_sim = cosine_similarity(embedding_matrix)

# %%
# Calculate the item-based predictions for positive interactions
item_EMBED_prediction = item_based_predict(entire_data, embedding_sim)
print("Predicted Interaction Matrix:")
print(item_EMBED_prediction)
print(item_EMBED_prediction.shape)

# %%
# CHECK PRECISION & RECALL NOT YET WITH CROSS-VALIDATION [OVERFITTING PROBLEM THOUGH]
precision_item_k, recall_item_k = precision_recall_at_k(item_EMBED_prediction, test_data_matrix, k=10)
print('Item-based EMBED Precision@K:', precision_item_k)
print('Item-based EMBED Recall@K:', recall_item_k)

# %%
#Cross Validation
def evaluate_one(seeds):
    train_df, test_df = random_split_per_user(interactions)
    train_matrix = create_data_matrix(train_df, n_users, n_items)

    # Compute similarity from current train split
    item_sim = cosine_similarity(train_matrix.T)
    prediction_matrix = item_based_predict(train_matrix, item_sim)

    # Evaluate on corresponding test set
    test_matrix = create_data_matrix(test_df, n_users, n_items)
    p_at_k, r_at_k = precision_recall_at_k(prediction_matrix, test_matrix, k=10)

    return p_at_k, r_at_k

# Run cross-validation
seeds = list(range(5))
results = Parallel(n_jobs=-1)(
    delayed(evaluate_one)(seed) for seed in seeds
)

# Unpack and average
precisions, recalls = zip(*results)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)

# Print results
print(f"Mean Precision@10 = {mean_precision:.4f}")
print(f"Mean Recall@10    = {mean_recall:.4f}")

# %% [markdown]
# ### BERT Similarity
# Mean Precision@10 = 0.0272 <br>
# Mean Recall@10    = 0.1760

# %%
# STEP 1: Combine text features
text_fields = ['Publisher', 'Subjects', 'google_api_title', 'author_clean', 'ISBN']
items['combined_text'] = items[text_fields].fillna('').agg(' '.join, axis=1)

# STEP 2: Align with train_data_matrix
items_ordered = items.set_index('book_id').loc[range(train_data_matrix.shape[1])]

# STEP 3: Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# STEP 4: Encode book texts into embeddings
bert_embeddings = model.encode(items_ordered['combined_text'].tolist(), show_progress_bar=True)

# STEP 5: Compute cosine similarity
bert_sim = cosine_similarity(bert_embeddings)

# %%
# Calculate the item-based predictions for positive interactions
item_bert_prediction = item_based_predict(train_data_matrix, bert_sim)
print("Predicted Interaction Matrix:")
print(item_bert_prediction)
print(item_bert_prediction.shape)

# %%
# Create recommendation
item_bert_recommendations_df = create_recommendation_table(item_bert_prediction, top_n=10, separator=" ")

# Save and display
item_bert_recommendations_df.to_csv('item_bert_recommendations.csv', index=False)

print("\nItem-based Recommendations:")
display(item_bert_recommendations_df)

# %%
p_at_k, r_at_k = precision_recall_at_k(item_bert_prediction, test_data_matrix, k=10)
print(f"Precision@10 = {p_at_k:.4f}")
print(f"Recall@10 = {r_at_k:.4f}")

# %%
# Cross-validation setup
seeds = list(range(5))  # 5 random seeds for 5 train-test splits

# Evaluate precision and recall for one run
def evaluate_one(seed):
    train_df, test_df = random_split_per_user(interactions, seed=seed)
    train_matrix = create_data_matrix(train_df, n_users, n_items)
    prediction_matrix = item_based_predict(train_matrix, bert_sim)
    test_matrix = create_data_matrix(test_df, n_users, n_items)
    p_at_k, r_at_k = precision_recall_at_k(prediction_matrix, test_matrix, k=10)
    return p_at_k, r_at_k

# Run evaluations in parallel
results = Parallel(n_jobs=-1)(
    delayed(evaluate_one)(seed) for seed in seeds
)

# Extract and average
precisions, recalls = zip(*results)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)

# Print results
print(f"Mean Precision@10 = {mean_precision:.4f}")
print(f"Mean Recall@10    = {mean_recall:.4f}")

# %% [markdown]
# ## Colaborative Filtering

# %% [markdown]
# ### CF Item-based
# Mean Precision@10 = 0.0585 <br>
# Mean Recall@10    = 0.2823

# %%
# Compute the item-item similarity matrix
item_similarity = cosine_similarity(entire_data.T)
print("Item-Item Similarity Matrix:")
print(item_similarity)
print(item_similarity.shape)

# %%
# Calculate the item-based predictions for positive interactions
item_prediction = item_based_predict(entire_data, item_similarity)
print("Predicted Interaction Matrix:")
print(item_prediction)
print(item_prediction.shape)

# %%
# Create recommendation
item_CF_recommendations_df = create_recommendation_table(item_prediction, top_n=10, separator=" ")

# Save and display
item_CF_recommendations_df.to_csv('item_CF_recommendations.csv', index=False)

print("\nItem-based Recommendations:")
display(item_CF_recommendations_df)

# %%
p_at_k, r_at_k = precision_recall_at_k(item_prediction, test_data_matrix, k=10)
print(f"Precision@10 = {p_at_k:.4f}")
print(f"Recall@10 = {r_at_k:.4f}")

# %%
# Cross-validation setup
seeds = list(range(5))  # 5 random seeds for 5 train-test splits

# Evaluate precision and recall for one run
def evaluate_one(seed):
    train_df, test_df = random_split_per_user(interactions, seed=seed)
    train_matrix = create_data_matrix(train_df, n_users, n_items)
    prediction_matrix = item_based_predict(train_matrix, bert_sim)
    test_matrix = create_data_matrix(test_df, n_users, n_items)
    p_at_k, r_at_k = precision_recall_at_k(prediction_matrix, test_matrix, k=10)
    return p_at_k, r_at_k

# Run evaluations in parallel
results = Parallel(n_jobs=-1)(
    delayed(evaluate_one)(seed) for seed in seeds
)

# Extract and average
precisions, recalls = zip(*results)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)

# Print results
print(f"Mean Precision@10 = {mean_precision:.4f}")
print(f"Mean Recall@10    = {mean_recall:.4f}")

# %% [markdown]
# ### CF User-based
# Mean Precision@10 = 0.0612 <br>
# Mean Recall@10    = 0.3167

# %%
# Compute the user-user similarity matrix
user_similarity = cosine_similarity(entire_data)
print("User-User Similarity Matrix:")
print(user_similarity)

# Check the shape as a sanity check
print("Shape of User Similarity Matrix:", user_similarity.shape)

# %%
# Define the function to predict interactions based on user similarity
def user_based_predict(interactions, similarity, epsilon=1e-9):
    """
    Predicts user-item interactions based on user-user similarity.
    Parameters:
        interactions (numpy array): The user-item interaction matrix.
        similarity (numpy array): The user-user similarity matrix.
        epsilon (float): Small constant added to the denominator to avoid division by zero.
    Returns:
        numpy array: The predicted interaction scores for each user-item pair.
    """
    # Calculate the weighted sum of interactions based on user similarity
    pred = similarity.dot(interactions) / (np.abs(similarity).sum(axis=1)[:, np.newaxis] + epsilon)
    return pred

# Calculate the user-based predictions for positive interactions
user_prediction = user_based_predict(entire_data, user_similarity)
print("Predicted Interaction Matrix (User-Based):")
print(user_prediction)
print(user_prediction.shape)

# %%
# Create recommendation
user_CF_recommendations_df = create_recommendation_table(user_prediction, top_n=10, separator=" ")

# Save and display
user_CF_recommendations_df.to_csv('user_CF_recommendations.csv', index=False)

print("\nuser-based Recommendations:")
display(user_CF_recommendations_df)

# %%
p_at_k, r_at_k = precision_recall_at_k(user_prediction, test_data_matrix, k=10)
print(f"Precision@10 = {p_at_k:.4f}")
print(f"Recall@10 = {r_at_k:.4f}")

# %%
#Cross Validation
def evaluate_one(seed):
    train_df, test_df = random_split_per_user(interactions, seed=seed)
    train_matrix = create_data_matrix(train_df, n_users, n_items)

    # Compute similarity from current train split
    user_sim = cosine_similarity(train_matrix)
    prediction_matrix = user_based_predict(train_matrix, user_sim)

    # Evaluate on corresponding test set
    test_matrix = create_data_matrix(test_df, n_users, n_items)
    p_at_k, r_at_k = precision_recall_at_k(prediction_matrix, test_matrix, k=10)

    return p_at_k, r_at_k

# Run cross-validation
seeds = list(range(5))
results = Parallel(n_jobs=-1)(
    delayed(evaluate_one)(seed) for seed in seeds
)

# Unpack and average
precisions, recalls = zip(*results)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)

# Print results
print(f"Mean Precision@10 = {mean_precision:.4f}")
print(f"Mean Recall@10    = {mean_recall:.4f}")


