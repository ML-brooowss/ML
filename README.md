# ML
Kaggle competition - git repository 
🚚 DELIVERABLES

(One). A project GitHub page. The readme.md is your report. There, in addition to other things, report the following table. Do hyper-parameter optimization to find the best solution. Your code should justify your results. (Note, to fill in this table, you have to use the train test and cross-validation, since only there you know the true labels).

user-user CF	item-item CF	Any other technique
Precision@10			
Recall@10			
* Report your Exploratory Data Analysis (EDA) on the interactions data and the items metadata.

## 📊 Datasets Overview

| Metric              | Count   |
|---------------------|--------:|
| Total interactions  | 87,047  |
| Unique items        | 15,291  |
| Unique users        | 7,838   |

Let's look at the data we have for the interactions.
![Non-missing values per source](readme_images/distribution_interactions_per_user.png)

Another couple of key metrics:
* The average number of interactions between a user and books is 11
* The median number of interactions is 6

We see that the distribution of interactions are positively skewed, with users having up to 385 interactions with reading materials!

Let's now look at the data we have for the items. The first thing to look at is how complete our dataset is, or how much missing data it has. 

![Non-missing values per source](readme_images/non_missing_data_plot.png)

<iframe src="topic_map_full.html" width="100%" height="600"></iframe>

* Which is the best model?
* Show examples of recommendations for some users. Do they align with the users' history of book rentals? Report some examples of “good” predictions, and some "bad" predictions. Do they make sense?
* Use data augmentation. There exist several APIs (eg Google Books or ISBNDB) that bring extra data using the ISBN of a book. Additionally, you may use the metadata available for the items (books).
Have a position on the leaderboard of this competition, with score better than 0.1452.

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Book Recommendation System: Final Report</title>
</head>
<body>
  <h1>Book Recommendation System: Final Report</h1>

  <h2>Overview</h2>
  <p>This project aims to develop a recommender system that proposes books to users based on either their previous behavior (interaction history) or the characteristics of the books themselves. We explored three main approaches:</p>
  <ol>
    <li><strong>Collaborative Filtering</strong>: Based on user-item interaction patterns.</li>
    <li><strong>Content-Based Filtering</strong>: Based on item attributes like title, genre, and description.</li>
    <li><strong>Hybrid Recommender</strong>: A combination of both approaches.</li>
  </ol>
  <p>We evaluated different models using metrics like <strong>Precision@K</strong> and <strong>Recall@10</strong> to measure the effectiveness of recommendations.</p>

  <h2>1. Collaborative Filtering (CF)</h2>
  <p>Collaborative filtering makes recommendations by analyzing past user behavior (e.g., which books were read) and identifying similarities between users or items.</p>

  <h3>1.1 User-Based CF</h3>
  <ul>
    <li><strong>Concept</strong>: Recommend books liked by users who are similar to the target user.</li>
    <li><strong>Baseline similarity</strong>: Cosine similarity
      <ul>
        <li>Measures the angle between item vectors; suitable for sparse, implicit data.</li>
      </ul>
    </li>
    <li><strong>K-Nearest Neighbors (KNN)</strong>:
      <ul>
        <li>We tested different values for k (number of neighbors) and found optimal performance at <strong>k = 70</strong></li>
        <li><em>[insert graph]</em></li>
      </ul>
    </li>
  </ul>
  <p><strong>Conclusion</strong>: Cosine similarity consistently outperformed other metrics for item-item collaborative filtering in our implicit feedback setting.</p>

  <h3>1.2 Item-Based CF</h3>
  <ul>
    <li><strong>Concept</strong>: Recommend books similar to those a user already interacted with.</li>
    <li><strong>Baseline Similarity</strong>: Cosine similarity
      <ul>
        <li>Measures the angle between item vectors; suitable for sparse, implicit data.</li>
      </ul>
    </li>
    <li><strong>K-Nearest Neighbors (KNN)</strong>:
      <ul>
        <li>We tested different values for k (number of neighbors) and found optimal performance at <strong>k = 70</strong></li>
        <li><em>[insert graph]</em></li>
      </ul>
    </li>
    <li><strong>Pearson Correlation</strong>: Not used because it's more effective for <strong>explicit ratings</strong>, that is when customers give explicit ratings (e.g., from 1–5). Pearson correlation adjusts for user bias.</li>
  </ul>
  <p><strong>Conclusion</strong>: Cosine similarity consistently outperformed other metrics for item-item collaborative filtering in our implicit feedback setting <a href="https://link.springer.com/chapter/10.1007/978-981-10-7398-4_37" target="_blank">[in line with academic literature]</a>.</p>

  <h2>2. Content-Based Filtering (CBF)</h2>
  <p>Content-based filtering recommends books that are similar in content to those the user liked previously. This method does not depend on what other users did. To compare book content, we transformed textual metadata (title, author, description, etc.) into <strong>embeddings</strong>: numerical vector representations of the semantic meaning of a piece of text (e.g., author) that allow us to compute similarity. This allows us to calculate similarity scores using methods like cosine similarity.</p>

  <h3>2.2 Embedding Techniques Used</h3>

  <h4>TF-IDF (Term Frequency-Inverse Document Frequency)</h4>
  <ul>
    <li><strong>What</strong>: A classic method in information retrieval. TF-IDF breaks down text into individual tokens and measures how important each word is in a book’s metadata relative to all other books.</li>
    <li><strong>How</strong>: Represents text as sparse vectors based on word frequency, adjusted by how unique each word is across the dataset.</li>
    <li><strong>Use Case</strong>: Good for surface-level textual similarities (e.g., shared keywords).</li>
    <li><strong>Example</strong>: We have a book with Title = <em>Harry Potter and the Philosopher's Stone</em>, Author = <em>J.K. Rowling</em>, Publisher = <em>Bloomsbury</em>. We concatenate all the metadata into one string: “Harry Potter and the Philosopher's Stone J.K. Rowling Bloomsbury”. TF-IDF counts the frequency of each word, downweights common words like “publishing”, and generates a sparse vector. If two books share the same publisher or author, they will appear similar based on the sparse vector.</li>
  </ul>

  <h4>BERT Embeddings</h4>
  <ul>
    <li><strong>What</strong>: Deep learning model by Google (transformer architecture) that takes full phrases or sentences (not just individual words like TF-IDF). It understands the context and relationships between words using a transformer model and outputs a dense vector that encodes the overall semantic meaning.</li>
    <li><strong>How</strong>: Generates dense, contextualized embeddings that understand the semantic meaning of sentences.</li>
    <li><strong>Use Case</strong>: Captures deeper relationships in content (e.g., plot similarities).</li>
    <li><strong>Example</strong>: Again, we create a concatenated string: “Harry Potter and the Philosopher's Stone J.K. Rowling Bloomsbury”. Instead of counting each word, BERT processes the entire sentence and understands that “Harry Potter and the Philosopher's Stone” is a title, “J.K. Rowling” is an author, and “Bloomsbury” is an organization. It doesn’t rely on exact matches and can understand that “Bloomsbury” and “BiggerPockets” are both major publishers.</li>
  </ul>

  <h4>Google Embedding API</h4>
  <ul>
    <li><strong>What</strong>: A cloud-based API that produces semantic embeddings from text.</li>
    <li><strong>How</strong>: Uses pretrained large language models (similar to BERT).</li>
    <li><strong>Use Case</strong>: Easy to integrate and computationally efficient.</li>
  </ul>

  <h2>3. Hybrid Recommender System</h2>
  <p>We combined both collaborative and content-based approaches using a <strong>weighted sum</strong> of different similarity matrices.</p>

  <pre><code>hybrid_sim = a * tfidf_sim + b * item_cf_sim + c * google_sim + d * bert_sim</code></pre>

  <p>We did not perform full grid search and/or cross-validation due to computational limits, but used a <strong>simplified tuning</strong> to demonstrate the concept.</p>
  <p>We found the highest precision using BERT, Google, and item-CF. The result of the simplified grid search showed:</p>
  <p><em>[Insert table: highlight best combo]</em></p>

  <p><strong>Note</strong>: Without cross-validation, results may overfit. However, the hybrid approach still showed the best overall performance in our simplified tests.</p>
</body>
</html>


<table border="1" style="border-collapse: collapse; text-align: center; width: 100%;">
  <thead style="background-color: #f5f5f5;">
    <tr>
      <th></th>
      <th><strong>user-user CF</strong></th>
      <th><strong>item-item CF</strong></th>
      <th><strong>BERT (item-based)</strong></th>
      <th><strong>TF-IDF (item-based)</strong></th>
      <th><strong>Google API (item-based)</strong></th>
      <th><strong>Hybrid (CF + Content + Popularity)</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Precision@10</strong></td>
      <td><!-- your value --></td>
      <td><!-- your value --></td>
      <td><!-- your value --></td>
      <td><!-- your value --></td>
      <td><!-- your value --></td>
      <td><!-- your value --></td>
    </tr>
    <tr>
      <td><strong>Recall@10</strong></td>
      <td><!-- your value --></td>
      <td><!-- your value --></td>
      <td><!-- your value --></td>
      <td><!-- your value --></td>
      <td><!-- your value --></td>
      <td><!-- your value --></td>
    </tr>
  </tbody>
</table>


