# ML
Kaggle competition - git repository 
🚚 DELIVERABLES

(One). A project GitHub page. The readme.md is your report. There, in addition to other things, report the following table. Do hyper-parameter optimization to find the best solution. Your code should justify your results. (Note, to fill in this table, you have to use the train test and cross-validation, since only there you know the true labels).

user-user CF	item-item CF	Any other technique
Precision@10			
Recall@10			
* Report your Exploratory Data Analysis (EDA) on the interactions data and the items metadata.
* Which is the best model?
* Show examples of recommendations for some users. Do they align with the users' history of book rentals? Report some examples of “good” predictions, and some "bad" predictions. Do they make sense?
* Use data augmentation. There exist several APIs (eg Google Books or ISBNDB) that bring extra data using the ISBN of a book. Additionally, you may use the metadata available for the items (books).
Have a position on the leaderboard of this competition, with score better than 0.1452.

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


