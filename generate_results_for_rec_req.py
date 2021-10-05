import os
import parse
import pandas as pd

base_path = 'rec-req_raw_data/'
rows = []
for file_name in os.listdir(base_path):
    topic_id = parse.parse('{}.txt',file_name)[0]
    docs = set()
    with open(os.path.join(base_path, file_name)) as result_file:
        for line in result_file.readlines():
            rel, doc_id, text = line.split('\t')
            key = topic_id + doc_id
            if key not in docs:
                rows.append((topic_id, doc_id, rel, 'rec-req'))
                docs.add(key)
pass
df = pd.DataFrame(rows)
df.to_csv('rec_req_result.txt', header=None, index=False, sep=' ')
# rows = []
# file_name = 'trec_hill_black_box_1_search_10_iters/trec_eval_search_30_iter_30_keywords_30'
# with open('{}.txt'.format(file_name)) as f:
#     docs = set()
#     for line in f.readlines():
#         topic_id, doc_id, score, run_name = line.split(' ')
#         key = topic_id + doc_id
#         if key not in docs:
#             rows.append([topic_id, doc_id, score, 'hill_black_box'])
#             docs.add(key)
# pd.DataFrame(rows).to_csv('{}_fixed.txt'.format(file_name), header=None, index=False, sep=' ')