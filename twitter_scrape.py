import twint
import pandas as pd


def find_url(tweet):
    index = tweet.find("http")
    if index != -1:
        return tweet[:index].strip()
    else:
        return tweet


def find_end_first_url(url):
    if url is not None:
        parts = url.split()
        return parts[0]


def run_twint():
    c = twint.Config()
    c.Username = "Reuters"
    # c.Links = "include"
    c.Pandas = True
    c.Limit = 1000
    c.Store_csv = True
    c.Output = "temp.csv"
    # c.Year = "2020"
    c.Since = "2020-06-01"
    c.Until = "2020-07-30"

    twint.run.Search(c)


# run_twint()

csv_file = "temp.csv"
df = pd.read_csv(csv_file, quotechar='"', skipinitialspace=True, header=None, dtype='string', index_col=None)
print(df.head())

new_df = pd.DataFrame()
new_df['full_title'] = df.iloc[:, 10]
# new_df = df.iloc[:, 10]
new_df['date'] = df.iloc[:, 3]


new_df['date'] = new_df['date'].str.slice(0, 4)
new_df['full_title'] = new_df['full_title'].apply(find_url)
# new_df = new_df.apply(find_url)
print(new_df)


# new_df = new_df.apply(find_end_first_url)
new_df.dropna(inplace=True)
print(new_df)

new_df.to_csv(path_or_buf="data/urls.csv", index=False)
