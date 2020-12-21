import twint
import pandas as pd


YEAR = "2020"
TWITTER_USERNAME = "Reuters"


def find_url(tweet):
    index = tweet.find("http")
    if index != -1:
        return tweet[:index].strip()
    else:
        return tweet


def run_twint():
    c = twint.Config()
    c.Username = TWITTER_USERNAME
    c.Links = "include"
    c.Pandas = True
    c.Limit = 1000
    c.Store_csv = True
    c.Output = "raw" + YEAR + TWITTER_USERNAME + ".csv"
    # c.Year = "2020"
    c.Since = YEAR + "-06-01"
    c.Until = YEAR + "-07-30"

    twint.run.Search(c)


run_twint()

csv_file = "raw" + YEAR + TWITTER_USERNAME + ".csv"
df = pd.read_csv(csv_file, quotechar='"', skipinitialspace=True, dtype='string', index_col=None)

new_df = pd.DataFrame()
new_df['full_title'] = df.iloc[:, 10]
new_df['date'] = df.iloc[:, 3]


new_df['date'] = new_df['date'].str.slice(0, 4)
new_df['full_title'] = new_df['full_title'].apply(find_url)
print(new_df)

new_df.dropna(subset=['full_title'], inplace=True)  # removes duplicates

new_df.to_csv(path_or_buf="data/" + YEAR + TWITTER_USERNAME + ".csv", index=False)
