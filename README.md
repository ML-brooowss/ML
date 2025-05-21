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
![Distribution of interactions per user](readme_images/distribution_interactions_per_user.png)

Another couple of key metrics:
* The average number of interactions between a user and books is 11
* The median number of interactions is 6

We see that the distribution of interactions are positively skewed, with users having up to 385 interactions with reading materials!

Let's now look at the data we have for the items. The first thing to do was to perform a bit of data cleaning by:
* Extracting the first valid ISBN from the 'ISBN Valid' column 
* Cleaning the titles as they had a trailing '/' and to support our data enhancing
* Cleaning the authors from birth and death years to ensure a consistent data formatting.

The next step was to look at is how complete our dataset is, or how much missing data it has. 

![Non-missing values per source](readme_images/non_missing_data_plot.png)

We see that the only datapoints that we consistently have throughout all the items is the index (i) and the title, and publishers for almost all of them. Otherwise, roughly 5% of the ISBNs, 15% Authors and 17% of Subjects are missing.

There are also many other data points that we don't have: language, book description, publication date and perhaps information about the book covers. 

How could we possibly remedy this and enhance the data we have access to?

## Data enhancing 

### Google Books API

The data enhancing for this step was composed of two parts. First, looking up books based on their first valid ISBN to extract the following missing entries:
* Book Description
* Publisher
* Subjects

As well as new data points:
* Google's Canonical Link, or the permanent link to the Google Books entry of the book. This will possibly useful for our UI later.
* Google's Image Link, or the permanent link to the book's cover. This will also possibly useful for our UI later.
* Language of the book, possibly useful for our embeddings later.
* Publication date of the book, possibly useful for our embeddings later.

Second, we also looked up books by their title to try to extract their ISBN as well as all the other datapoints mentioned above. In doing so, we enhance the potential entries we find using our second data enhancing method. 

### ISBN Database API

The same datapoints as above were extracted using the ISBN Database. We combined all these newly found datapoints in the following way: first priority for all the fields to the original dataframe, then Google API entries, then ISBN Database entries. An exception to that is for the Image Link, which turned out being fallacious for many entries. We therefore gave priority to the Image link provided by the ISBN Database. When running our models, we tried giving the opposite priority to the two enhanced database, which did not change results by much. However, we made the choice to keep the entries from the original dataset intact, considering that it is the ground truth.

The final results of our data enhancing techniques are shown in the figure here below. The light blue bar indicates the original dataset, the next two bars the results of the individual data enhancing techniques and the last bar indicating the resulting dataframe after the combination of both methods. We're able to achieve remarkable results across all dimensions, hitting almost 80% and above for all the data points. 

![Final non missing data](readme_images/non_missing_data_plot.png)

## Data enhancing extension

In addition to the previous enhancing methods, we used BERTopics to extract the topics for each document from our corpus. To do so, we used the title and the description of each book. First, the algorithm uses a pretrained BERT model to capture semantic meaning of the text down to its core. Then, we used the built-in UMAP dimensionality reduction function to cluster the topics into 25 topics. The results are as follow:

![Reduced topic distributions](readme_images/reduced_topic_distribution.png)

We see that the large majority of the topics are unidentified by the model. The most prominent topics seem to be feminism, psychology and academic research. We could have manually labelled the clusters to make them more human-friendly, but decided to keep them as such. In fact, we used these topics for our recommender system as they were later on using embeddings, as seen in the next section. An interesting extension to our work would be to run cross validation to find the optimal number of topics for the embeddings.

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


