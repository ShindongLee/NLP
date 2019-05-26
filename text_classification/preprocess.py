import csv
import time
from bisect import bisect_left

#sorted list 에서 item idx 찾기 O(logN)
def binary_search(list, item):
    item_index = bisect_left(list, item)
    assert item_index != len(list) and list[item_index] == item
    return item_index
    
#word -> hash
def fnv1a_hash(word, max_entries = int(2.10 * (10**6)), basis = 2166136261, prime = 16777619):
    hash = basis
    for char in word:
        char2num = ord(char)
        hash ^= char2num
        hash *= prime
        hash %= max_entries
    return hash
    
#train.csv 읽어서 training data 만들기
def read_data(use_bigram):
    start_time = time.time()

    label = []
    train_set_word = []
    train_set_idx = []
    word_set = set()
    
    with open('./ag_news_csv/train.csv') as train_csv:
        fullcsv = csv.reader(train_csv, delimiter = ',')
        for line in fullcsv:
            answer = int(line[0]) - 1 # 0, 1, 2, 3
            title = line[1]
            body = line[2]
            title_body = title + ' ' + body
            word_list = title_body.split()
            label.append(answer)
            if use_bigram == 'False':
                train_set_word.append(word_list)
                for word in word_list:
                    word_set.add(word)
            elif use_bigram == 'True':
                bigram_lst = []
                for word_idx in range(len(word_list) - 1):
                    bigram = word_list[word_idx] + '-' + word_list[word_idx + 1]
                    word_set.add(bigram)
                    bigram_lst.append(bigram)
                train_set_word.append(bigram_lst)
        print('Number of words(bigrams) before hashing:', len(word_set))
        
        # word to hash
        w2h = {}        
        for word in word_set:
            w2h[word] = fnv1a_hash(word)
        
        # word to consecutive index
        w2i = {}
        hash_lst = set()
        
        for word in w2h:
            hash_lst.add(w2h[word])    
        hash_lst = list(hash_lst)
        hash_lst.sort()
        
        print('Number of words(bigrams) after hashing:', len(hash_lst))
        
        for word in w2h:
            w2i[word] = binary_search(hash_lst, w2h[word])
        
        # train_set 을 list(list(word)) 에서 list(list(idx))로 바꿈
        for word_lst in train_set_word:
            idx_lst = []
            for word in word_lst:
                idx_lst.append(w2i[word])
            train_set_idx.append(idx_lst)
        
        assert len(train_set_idx) == len(label)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print('Time Elapsed for Preprocessing %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
        
        return train_set_idx, label, w2i
    
    