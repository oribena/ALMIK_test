import pandas as pd

judgment_df = pd.read_csv('adhoc-qrels_filtered', sep=' ', names=['topic', 'Q', 'tweet_id', 'relevance'])

res = judgment_df.groupby('topic')['relevance'].value_counts().unstack()
res.to_excel('judgment_statistics.xlsx')