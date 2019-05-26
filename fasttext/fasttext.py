import torch
import numpy as np
import time
import math
import random
import argparse
from random import shuffle
from collections import Counter
from bisect import bisect_left

torch.set_printoptions(threshold=10000)

def binary_search(list, item):
    item_index = bisect_left(list, item)
    assert item_index != len(list) and list[item_index] == item
    return item_index

def fnv1a_hash(gram, max_entries = int(2.10 * (10**6)), basis = 2166136261, prime = 16777619):
    hash = basis
    for char in gram:
        char2num = ord(char)
        hash ^= char2num
        hash *= prime
        hash %= max_entries
    return hash

def Analogical_Reasoning_Task(embedding, w2gi, g2i):
#######################  Input  #########################
# embedding : Word embedding (type:torch.tesnor(V,D))   #
#########################################################
    start_time = time.time()
    if type(embedding) is not torch.Tensor:
        embedding = torch.tensor(embedding)
    g = open("result.txt", 'w')
    total_question = 0.0
    total_correct = 0.0
    N = embedding.shape[0]
    
    vector = {}
    for word in w2gi:
        vector[word] = embedding[w2gi[word]].sum(0) / len(w2gi[word])
    
    l2_norm = {}
    for word in w2gi:
        l2_norm[word] = torch.dist(vector[word], torch.zeros_like(vector[word]), 2)
    
    input_words = ['narrow-mindedness', 'department', 'campfires', \
                    'knowing', 'urbanize', 'imperfection', 'principality', \
                    'abnormal', 'secondary', 'ungraceful']
    print('Input Words:', input_words)
    print()
    
    g.write("Input Words: \n")
    for input_word in input_words:
        g.write("%s\n" % input_word)
    g.write("\n")
    
    for input_word in input_words:
        input_grams = []
        similarity_top5 = [None, None, None, None, None]
        word2sim = {}
        word2sim[None] = -1.0
        input_grams.append('<'+input_word+'>')
        if len(input_word) >= 3:
            input_grams.append('<'+input_word[:2])
            input_grams.append(input_word[-2:]+'>')
            for idx3 in range(len(input_word) - 2):
                input_grams.append(input_word[idx3:idx3+3])
        if len(input_word) >= 4:
            input_grams.append('<'+input_word[:3])
            input_grams.append(input_word[-3:]+'>')
            for idx4 in range(len(input_word) - 3):
                input_grams.append(input_word[idx4:idx4+4])
        if len(input_word) >= 5:
            input_grams.append('<'+input_word[:4])
            input_grams.append(input_word[-4:]+'>')
            for idx5 in range(len(input_word) - 4):
                input_grams.append(input_word[idx5:idx5+5])
        if len(input_word) >= 6:
            input_grams.append('<'+input_word[:5])
            input_grams.append(input_word[-5:]+'>')
            for idx6 in range(len(input_word) - 5):
                input_grams.append(input_word[idx6:idx6+6])
         
        input_grams_idx_lst = []
        for input_gram in input_grams:
            if input_gram in g2i:
                input_grams_idx_lst.append(g2i[input_gram] + 1)
            
        input_vector = embedding[input_grams_idx_lst].sum(0) / len(input_grams_idx_lst)
        input_vecotr_l2_norm = torch.dist(input_vector, torch.zeros_like(input_vector), 2)
        
        for word in vector:
            similarity = torch.dot(vector[word], input_vector) / (l2_norm[word] * input_vecotr_l2_norm)
            word2sim[word] = similarity
            if similarity > word2sim[similarity_top5[0]]:
                similarity_top5[4] = similarity_top5[3]
                similarity_top5[3] = similarity_top5[2]
                similarity_top5[2] = similarity_top5[1]
                similarity_top5[1] = similarity_top5[0]
                similarity_top5[0] = word
            elif similarity > word2sim[similarity_top5[1]]:
                similarity_top5[4] = similarity_top5[3]
                similarity_top5[3] = similarity_top5[2]
                similarity_top5[2] = similarity_top5[1]
                similarity_top5[1] = word
            elif similarity > word2sim[similarity_top5[2]]:
                similarity_top5[4] = similarity_top5[3]
                similarity_top5[3] = similarity_top5[2]
                similarity_top5[2] = word
            elif similarity > word2sim[similarity_top5[3]]:
                similarity_top5[4] = similarity_top5[3]
                similarity_top5[3] = word
            elif similarity > word2sim[similarity_top5[4]]:
                similarity_top5[4] = word
        
        print('Top 5 words similar to word: ', input_word)
        print('1st : {:<25} with similarity : {:.4f}'.format(similarity_top5[0], word2sim[similarity_top5[0]]))
        print('2nd : {:<25} with similarity : {:.4f}'.format(similarity_top5[1], word2sim[similarity_top5[1]]))
        print('3rd : {:<25} with similarity : {:.4f}'.format(similarity_top5[2], word2sim[similarity_top5[2]]))
        print('4th : {:<25} with similarity : {:.4f}'.format(similarity_top5[3], word2sim[similarity_top5[3]]))
        print('5th : {:<25} with similarity : {:.4f}'.format(similarity_top5[4], word2sim[similarity_top5[4]]))
        print()
        
        g.write("Top 5 words similar to word: %s\n" % input_word)
        g.write("1st : {:<25} with similarity : {:.4f}\n".format(similarity_top5[0], word2sim[similarity_top5[0]]))
        g.write("2nd : {:<25} with similarity : {:.4f}\n".format(similarity_top5[1], word2sim[similarity_top5[1]]))
        g.write("3rd : {:<25} with similarity : {:.4f}\n".format(similarity_top5[2], word2sim[similarity_top5[2]]))
        g.write("4th : {:<25} with similarity : {:.4f}\n".format(similarity_top5[3], word2sim[similarity_top5[3]]))
        g.write("5th : {:<25} with similarity : {:.4f}\n".format(similarity_top5[4], word2sim[similarity_top5[4]]))
        g.write("\n")
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    g.write("Time Elapsed for Validaiton %02d:%02d:%02d\n" % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    print('Time Elapsed for Validaiton %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

def subsampling(input_seq, target_seq, discard_prob, thresh = 0.00001):
###############################  Output  #########################################
# subsampled : Subsampled sequence                                               #
##################################################################################
    subsampling_start_time = time.time()

    remove_list = []
    input_seq_ssp = []
    target_seq_ssp = []
    discard = 0

    for i_idx in range(len(input_seq)):
        prob = discard_prob[input_seq[i_idx]]
        f = np.random.rand()
        if f >= prob:
            input_seq_ssp.append(input_seq[i_idx])
            target_seq_ssp.append(target_seq[i_idx])
        else:
            discard += 1

    subsampling_end_time = time.time()
    time_elapsed = subsampling_end_time - subsampling_start_time
    print('Time Elapsed for Subsampling %02d:%02d:%02d\n' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    print('Training Set before Subsampling :', len(input_seq))                
    print('Number of discard :', discard) 
    print('Training Set after Subsampling :', len(input_seq_ssp))

    return input_seq_ssp, target_seq_ssp 

def subword(CentorWordgrams, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################
    
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]
    C = len(CentorWordgrams)
    K = outputMatrix.shape[0]

    hidden_layer = inputMatrix[CentorWordgrams].sum(0).view(1,D) / C
    output_layer = torch.mm(hidden_layer, outputMatrix.t()) # [1 D] X [D K] = [1 K]

    target = output_layer[0][0]
    NS_samples = output_layer[0][1:]
    
    loss = -torch.log(torch.sigmoid(target)) - torch.sum(torch.log(torch.sigmoid(-NS_samples)))
    #loss = -torch.sum(torch.log(torch.sigmoid(mul)))
    #print(loss)

    grad_output_layer = torch.sigmoid(output_layer) #[1 K]
    
    grad_output_layer[0][0] -= 1.0 #[1 K]
    
    grad_out = torch.mm(grad_output_layer.t(), hidden_layer) #[K 1] X [1 D] = [K D]
    grad_in = torch.mm(grad_output_layer, outputMatrix) #[1 K] X [K D] = [1 D]
    
    return loss, grad_in, grad_out

def word2vec_trainer(input_seq, target_seq, numwords, numgrams, w2gi, i2w, stats, frequency, use_subsampling = True, threshold = 0.00001, NS=20, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numgrams, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()
    
    discard_prob = {}
    for word_id in frequency:
        discard_prob[word_id] = max(0, 1 - math.sqrt(threshold / frequency[word_id]))

    for _ in range(epoch):
        #Training word2vec using SGD(Batch size : 1)
        
        if use_subsampling:
            input_seq_ssp, target_seq_ssp = subsampling(input_seq, target_seq, discard_prob, thresh = 0.00001)
        else:
            input_seq_ssp, target_seq_ssp = input_seq, target_seq

        for inputs, output in zip(input_seq_ssp,target_seq_ssp):
            i+=1
            NS_selected = random.sample(stats, NS)
            while output in NS_selected:
                NS_selected.remove(output)
            activated = [output] + NS_selected
            L, G_in, G_out = subword(w2gi[i2w[inputs]], W_in, W_out[activated])
            W_in[w2gi[i2w[inputs]]] -= learning_rate*G_in.squeeze()
            W_out[activated] -= learning_rate*G_out

            losses.append(L.item())
            #print(i)
            if i%50000==0:
            	avg_loss=sum(losses)/len(losses)
            	print("Loss : %f" %(avg_loss,))
            	losses=[]

    return W_in, W_out


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    part = args.part
    ns = args.ns

	#Load and preprocess corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("preprocessing...")
    corpus = text.split()
    stats = Counter(corpus)
    words = []
    grams = []
    #Discard rare words
    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)
    
    #Give an index number to a word
    w2i = {}
    w2i[" "]=0
    i = 1
    for word in vocab:
        w2i[word] = i
        i+=1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k
    
    #making n grams
    for word in vocab:
        grams.append('<'+word+'>')
        if len(word) >= 3:
            grams.append('<'+word[:2])
            grams.append(word[-2:]+'>')
            for idx3 in range(len(word) - 2):
                grams.append(word[idx3:idx3+3])
        if len(word) >= 4:
            grams.append('<'+word[:3])
            grams.append(word[-3:]+'>')
            for idx4 in range(len(word) - 3):
                grams.append(word[idx4:idx4+4])    
        if len(word) >= 5:
            grams.append('<'+word[:4])
            grams.append(word[-4:]+'>')
            for idx5 in range(len(word) - 4):
                grams.append(word[idx5:idx5+5])
        if len(word) >= 6:
            grams.append('<'+word[:5])
            grams.append(word[-5:]+'>')
            for idx6 in range(len(word) - 5):
                grams.append(word[idx6:idx6+6])

    #print(grams)
    grams = set(grams) #중복제거
    grams = list(grams)
    
    print('Gram Size before hashing : ', len(grams) + 1)
    
    #gram to hash(gram)
    #g2h : gram -> hash(gram)
    g2h = {}
    for gram in grams:
        g2h[gram] = fnv1a_hash(gram)
    
    #making hash consecutive (removing indices not used)
    #g2i : gram -> index of hash(gram) ex) 0, 1, 2, ... (consecutive)
    g2i = {}
    hash_lst = set()
    for gram in g2h:
        hash_lst.add(g2h[gram])
    hash_lst = list(hash_lst)
    hash_lst.sort()
    for gram in g2h:
        g2i[gram] = binary_search(hash_lst, g2h[gram])

    print('Gram Size after hashing : ', len(hash_lst) + 1)
    
    #w2gi : word -> list of index of grams in word
    w2gi = {}
    w2gi[" "] = [0]
    for word in vocab:
        gram_index_lst = []
        gram_index_lst.append(g2i['<'+word+'>']+1)
        if len(word) >= 3:
            gram_index_lst.append(g2i['<'+word[:2]] + 1)
            gram_index_lst.append(g2i[word[-2:]+'>'] + 1)
            for idx3 in range(len(word) - 2):
                gram_index_lst.append(g2i[word[idx3:idx3+3]] + 1)
        if len(word) >= 4:
            gram_index_lst.append(g2i['<'+word[:3]] + 1)
            gram_index_lst.append(g2i[word[-3:]+'>'] + 1)
            for idx4 in range(len(word) - 3):
                gram_index_lst.append(g2i[word[idx4:idx4+4]] + 1)
        if len(word) >= 5:
            gram_index_lst.append(g2i['<'+word[:4]] + 1)
            gram_index_lst.append(g2i[word[-4:]+'>'] + 1)
            for idx5 in range(len(word) - 4):
                gram_index_lst.append(g2i[word[idx5:idx5+5]] + 1)
        if len(word) >= 6:
            gram_index_lst.append(g2i['<'+word[:5]] + 1)
            gram_index_lst.append(g2i[word[-5:]+'>'] + 1)
            for idx6 in range(len(word) - 5):
                gram_index_lst.append(g2i[word[idx6:idx6+6]] + 1)
        
        w2gi[word] = gram_index_lst

    total_freq = 0
    #Code dict for hierarchical softmax
    freqdict={}
    freqdict[0]=10
    for word in vocab:
        freqdict[w2i[word]]=stats[word]
        total_freq += freqdict[w2i[word]]
    #codedict = HuffmanCoding().build(freqdict)
    #print(freqdict)
    
    frequency = {} #for subsampling
    for word_id in freqdict:
        frequency[word_id] = freqdict[word_id] / total_freq

    #Frequency table for negative sampling
    freqtable = [0,0,0]
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])
    
    #print(freqtable)

    #Make training set
    print("build training set...")
    #train_set = []
    
    input_set = []
    target_set = []
    window_size = 5

    for j in range(len(words)):
        if j<window_size:
            input_set += [w2i[words[j]] for _ in range(window_size*2)]
            target_set += [0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)]
        elif j>=len(words)-window_size:
            input_set += [w2i[words[j]] for _ in range(window_size*2)]
            target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)]
        else:
            input_set += [w2i[words[j]] for _ in range(window_size*2)]
            target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)]

    print("Vocabulary size")
    print(len(w2i))
    print()

    #Training section
    start_time = time.time()
    emb,_ = word2vec_trainer(input_set, target_set, len(w2i), len(grams) + 1, w2gi, i2w, freqtable, frequency, use_subsampling = True, threshold= 0.002, NS=ns, dimension=64, epoch=1, learning_rate=0.01)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Time Elapsed for Training %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    Analogical_Reasoning_Task(emb, w2gi, g2i)

main()