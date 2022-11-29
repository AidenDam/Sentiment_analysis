# from nltk.corpus import LazyCorpusLoader, WordListCorpusReader

# stopwords = LazyCorpusLoader(
#     "data", WordListCorpusReader, r"(?!README|\.).*", encoding="utf8", nltk_data_subdir=''
# )


from nltk.corpus import stopwords

print(len(set(stopwords.words('vietnamese-stopwords.txt'))))
# print(r"(?!README|\.).*")