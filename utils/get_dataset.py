# source: https://huggingface.co/datasets/uit-nlp/vietnamese_students_feedback

import requests
import csv
import pandas as pd
import os
import gdown
import shutil

from . import utils

path_nltk = os.path.join(os.path.expanduser('~'), 'nltk_data/corpora/stopwords/')
filename_stop_words = 'vietnamese-stopwords.txt'
filename_vocab = path_nltk + 'vocab.txt'

# Download data for Vietnamese
def init():
    if not os.path.exists(path_nltk):
        os.makedirs(path_nltk)

    if not os.path.exists(path_nltk + filename_stop_words):
        gdown.download('https://github.com/stopwords/vietnamese-stopwords/blob/master/vietnamese-stopwords.txt', path_nltk + filename_stop_words, quiet=False)
    
    if not os.path.exists(filename_vocab):
        gdown.download('https://drive.google.com/uc?id=1nzak5OkrheRV1ltOGCXkT671bmjODLhP&export=download', filename_vocab, quiet=False)
        
        words = utils.fil_voca(filename_vocab)
        with open(filename_vocab, 'w') as f:
            for word in words:
                f.write(f'{word}\n')
init()
# end


def imcomplete_version_datadet(filename_train, filename_val, filename_test):
    url_train = r'https://datasets-server.huggingface.co/first-rows?dataset=uit-nlp%2Fvietnamese_students_feedback&config=default&split=train'
    url_val = r'https://datasets-server.huggingface.co/first-rows?dataset=uit-nlp%2Fvietnamese_students_feedback&config=default&split=validation'
    url_test = r'https://datasets-server.huggingface.co/first-rows?dataset=uit-nlp%2Fvietnamese_students_feedback&config=default&split=test'

    def json2csv(url, out):
      request = requests.get(url)
      if request.status_code == 200:
        json = request.json()
        with open(out, 'w') as data:
          f = csv.writer(data)
          f.writerow(list(json['rows'][0]['row'].keys())[:-1])
          for it in json['rows']:
            f.writerow(list(it['row'].values())[:-1])

    json2csv(url_train, filename_train)
    json2csv(url_val, filename_val)
    json2csv(url_test, filename_test)

def complete_version_datadet(filename_train, filename_val, filename_test):
    if not os.path.exists('data/tmp'):
        os.mkdir('data/tmp')
    
    filename_down = (
        ("https://drive.google.com/uc?id=1nzak5OkrheRV1ltOGCXkT671bmjODLhP&export=download", "data/tmp/train_sentence.txt"),
        ("https://drive.google.com/uc?id=1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv&export=download", "data/tmp/train_sentiment.txt"),
        ("https://drive.google.com/uc?id=1sMJSR3oRfPc3fe1gK-V3W5F24tov_517&export=download", "data/tmp/val_sentence.txt"),
        ("https://drive.google.com/uc?id=1GiY1AOp41dLXIIkgES4422AuDwmbUseL&export=download", "data/tmp/val_sentiment.txt"),
        ("https://drive.google.com/uc?id=1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n&export=download", "data/tmp/test_sentence.txt"),
        ("https://drive.google.com/uc?id=1vkQS5gI0is4ACU58-AbWusnemw7KZNfO&export=download", "data/tmp/test_sentiment.txt")
    )

    for it in filename_down:
        gdown.download(it[0], it[1], quiet=False)

    def concat_data(url1, url2, out):
        with open(url1, 'r') as f:
            df1 = f.read().split('\n')[:-1]
        with open(url2, 'r') as f:
            df2 = f.read().split('\n')[:-1]
        
        df = pd.DataFrame({'sentence': df1,  'sentiment': df2})
        df.to_csv(out, index=False)

    concat_data('data/tmp/train_sentence.txt', 'data/tmp/train_sentiment.txt', filename_train)
    concat_data('data/tmp/val_sentence.txt', 'data/tmp/val_sentiment.txt', filename_val)
    concat_data('data/tmp/test_sentence.txt', 'data/tmp/test_sentiment.txt', filename_test)

    shutil.rmtree('data/tmp')

def get_dataset(data='full', check_down=False):
    filename_train = f'data/train_{data}.csv'
    filename_val = f'data/val_{data}.csv'
    filename_test = f'data/test_{data}.csv'

    if not os.path.exists('data'):
        os.mkdir('data')

    if check_down:
        if os.path.exists(filename_train) and os.path.exists(filename_test) and os.path.exists(filename_val):
            return filename_train, filename_val, filename_test

    if data == 'full':
        complete_version_datadet(filename_train, filename_val, filename_test)
    else:
        imcomplete_version_datadet(filename_train, filename_val, filename_test)
    
    return filename_train, filename_val, filename_test

if __name__ == '__main__':
    get_dataset(data='full', check_down=False)
