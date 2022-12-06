import pandas as pd
datasets = ['c', 'uc1', 'uc2', 'uc3', 'cu1', 'cu2', 'cu3']

#read the master datasets
df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
val_df =  pd.read_csv("validation.csv")
data_count = df.shape[0]

LABELS = 'train', 'test', 'val'
FOLDER = 'parsed_sets'

for ds in datasets:
    # if ds== 'c':
    #     for i, dfs  in enumerate([df, test_df, val_df]):
    #         new_df = dfs.copy()
    #         new_df = new_df.drop(['author', 'comment', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', "Unnamed: 0"], axis=1)
    #         #just rename to the labels we use for train
    #         new_df = new_df.rename(columns={'parent_comment': 'comment'})

    #         new_df.to_csv(f"{FOLDER}/{ds}_{LABELS[i]}.csv")

    if ds== 'uc3':
        print(f"parsing {ds}")
        for i, dfs  in enumerate([df, test_df, val_df]):
            print(f"PARSING {i}")
            new_df = dfs.copy()
            new_df = new_df.drop(['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', "Unnamed: 0"], axis=1)
            #just rename to the labels we use for train

            for j, (index, row) in enumerate(new_df.iterrows()):
                context = row['parent_comment']
                utterance = row['comment']
                cu = f"{utterance}</s>{context}"
                new_df.at[j,'comment']=cu

            new_df = new_df.drop(['parent_comment'], axis=1)
            new_df.to_csv(f"{FOLDER}/{ds}_{LABELS[i]}.csv")


    if ds== 'cu3':
        print(f"parsing {ds}")
        for i, dfs  in enumerate([df, test_df, val_df]):
            print(f"PARSING {i}")
            new_df = dfs.copy()
            new_df = new_df.drop(['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', "Unnamed: 0"], axis=1)
            #just rename to the labels we use for train

            for j, (index, row) in enumerate(new_df.iterrows()):
                context = row['parent_comment']
                utterance = row['comment']
                cu = f"{context}</s>{utterance}"
                new_df.at[j,'comment']=cu

            new_df = new_df.drop(['parent_comment'], axis=1)
            new_df.to_csv(f"{FOLDER}/{ds}_{LABELS[i]}.csv")


    if ds== 'uc1':
        print(f"parsing {ds}")
        for i, dfs  in enumerate([df, test_df, val_df]):
            print(f"PARSING {i}")
            new_df = dfs.copy()
            new_df = new_df.drop(['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', "Unnamed: 0"], axis=1)
            #just rename to the labels we use for train

            for j, (index, row) in enumerate(new_df.iterrows()):
                context = row['parent_comment']
                context_split = context.split(" ")
                context_len = len(context_split)
                cutoff = (1 * context_len)//3 
                utterance = row['comment']
                joined_cutoff = " ".join(context_split[:cutoff])
                cu = f"{utterance}</s>{joined_cutoff}"
                new_df.at[j,'comment']=cu

            new_df = new_df.drop(['parent_comment'], axis=1)
            new_df.to_csv(f"{FOLDER}/{ds}_{LABELS[i]}.csv")
    if ds== 'uc2':
        print(f"parsing {ds}")
        for i, dfs  in enumerate([df, test_df, val_df]):
            print(f"PARSING {i}")
            new_df = dfs.copy()
            new_df = new_df.drop(['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', "Unnamed: 0"], axis=1)
            #just rename to the labels we use for train

            for j, (index, row) in enumerate(new_df.iterrows()):
                context = row['parent_comment']
                context_split = context.split(" ")
                context_len = len(context_split)
                cutoff = (2 * context_len)//3 
                utterance = row['comment']
                joined_cutoff = " ".join(context_split[:cutoff])
                cu = f"{utterance}</s>{joined_cutoff}"
                new_df.at[j,'comment']=cu

            new_df = new_df.drop(['parent_comment'], axis=1)
            new_df.to_csv(f"{FOLDER}/{ds}_{LABELS[i]}.csv")
    if ds== 'cu1':
        print(f"parsing {ds}")
        for i, dfs  in enumerate([df, test_df, val_df]):
            print(f"PARSING {i}")
            new_df = dfs.copy()
            new_df = new_df.drop(['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', "Unnamed: 0"], axis=1)
            #just rename to the labels we use for train

            for j, (index, row) in enumerate(new_df.iterrows()):
                context = row['parent_comment']
                context_split = context.split(" ")
                context_len = len(context_split)
                cutoff = (1 * context_len)//3 
                utterance = row['comment']
                joined_cutoff = " ".join(context_split[:cutoff])
                cu = f"{joined_cutoff}</s>{utterance}"
                new_df.at[j,'comment']=cu

            new_df = new_df.drop(['parent_comment'], axis=1)
            new_df.to_csv(f"{FOLDER}/{ds}_{LABELS[i]}.csv")
    if ds== 'cu2':
        print(f"parsing {ds}")
        for i, dfs  in enumerate([df, test_df, val_df]):
            print(f"PARSING {i}")
            new_df = dfs.copy()
            new_df = new_df.drop(['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', "Unnamed: 0"], axis=1)
            #just rename to the labels we use for train

            for j, (index, row) in enumerate(new_df.iterrows()):
                context = row['parent_comment']
                context_split = context.split(" ")
                context_len = len(context_split)
                cutoff = (2 * context_len)//3 
                utterance = row['comment']
                joined_cutoff = " ".join(context_split[:cutoff])
                cu = f"{joined_cutoff}</s>{utterance}"
                new_df.at[j,'comment']=cu

            new_df = new_df.drop(['parent_comment'], axis=1)
            new_df.to_csv(f"{FOLDER}/{ds}_{LABELS[i]}.csv")





