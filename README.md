# ML
Kaggle competition - git repository 
🚚 DELIVERABLES

(One). A project GitHub page. The readme.md is your report. There, in addition to other things, report the following table. Do hyper-parameter optimization to find the best solution. Your code should justify your results. (Note, to fill in this table, you have to use the train test and cross-validation, since only there you know the true labels).

user-user CF	item-item CF	Any other technique
Precision@10			
Recall@10			
* Report your Exploratory Data Analysis (EDA) on the interactions data and the items metadata.

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Updated column names (renamed for clarity)
columns_of_interest = [
    'Image', 'Language', 'Published date', 'Subjects',
    'Authors', 'Title', 'Description', 'ISBN'
]


# Mapping for renaming columns
rename_mapping = {
    'PublishedDate': 'Published date',
    'author_clean': 'Authors',
    'title_clean': 'Title',
    'image': 'Image'
}

# Normalize 'ImageLink' to 'image'
def normalize_columns(df):
    df = df.copy()
    if 'ImageLink' in df.columns:
        df = df.rename(columns={'ImageLink': 'Image'})
    df = df.rename(columns=rename_mapping)
    # Remove duplicate columns, keeping the first occurrence
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# Normalize and subset safely
def safe_subset(df, columns):
    return df[[col for col in columns if col in df.columns]]

df_items = safe_subset(normalize_columns(items), columns_of_interest)
df_google = safe_subset(normalize_columns(google_enhanced_first), columns_of_interest)
df_isbn = safe_subset(normalize_columns(isbn_enhanced_first), columns_of_interest)

# Get all columns present for consistent plotting
all_cols = [col for col in columns_of_interest if col in df_items.columns or col in df_google.columns or col in df_isbn.columns]

priority_cols = ['Title', 'Authors', 'ISBN', 'Subjects']
remaining_cols = [col for col in all_cols if col not in priority_cols]
all_cols = priority_cols + remaining_cols

# Function to compute non-missing percentages
def get_non_missing_percentage(df):
    pct = df.notnull().sum() / len(df) * 100
    return pct.reindex(all_cols, fill_value=0)

pct_items = get_non_missing_percentage(df_items)
pct_google = get_non_missing_percentage(df_google)
pct_isbn = get_non_missing_percentage(df_isbn)

# Plotting
x = np.arange(len(all_cols))
width = 0.25

# Use Seaborn color palette for better aesthetics
palette = sns.color_palette("ch:s=.25,rot=-.25")  # Or try "deep", "muted", etc.

fig, ax = plt.subplots(figsize=(14, 6))

bars1 = ax.bar(x - width, pct_items, width, label='items', color=palette[0])
bars2 = ax.bar(x, pct_google, width, label='google_enhanced', color=palette[1])
bars3 = ax.bar(x + width, pct_isbn, width, label='isbn_enhanced', color=palette[2])

ax.set_ylabel('Non-Missing Percentage (%)')
ax.set_title('Non-Missing Data per Column Across Sources')
ax.set_xticks(x)
ax.set_xticklabels(all_cols, rotation=45, ha='right')
ax.set_ylim(0, 105)
ax.legend()

# Annotate bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(round(height))}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
````
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


