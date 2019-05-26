import torch
from random import shuffle
from collections import Counter
import argparse
from huffman import HuffmanCoding
import numpy as np
import time
import math
import random

torch.set_printoptions(threshold=10000)

def Analogical_Reasoning_Task(embedding, w2i, i2w):
#######################  Input  #########################
# embedding : Word embedding (type:torch.tesnor(V,D))   #
#########################################################
    start_time = time.time()
    embedding = torch.tensor(embedding)
    #print(w2i)
    f = open("questions-words.txt", 'r')
    g = open("result.txt", 'w')
    total_question = 0.0
    total_correct = 0.0
    N = embedding.shape[0]
    
    vector = {}
    for word in w2i:
        vector[word] = embedding[w2i[word]]
    
    l2_norm = {}
    for word in w2i:
        l2_norm[word] = torch.dist(vector[word], torch.zeros_like(vector[word]), 2)
    
    while True:
        line = f.readline()
        if not line: break
        if line[0] == ':': continue
        Qwords = line.split()
        if Qwords[0].lower() in w2i and Qwords[1].lower() in w2i and Qwords[2].lower() in w2i and Qwords[3].lower() in w2i:
            total_question += 1.0
            word_1_vec = vector[Qwords[0].lower()]
            word_2_vec = vector[Qwords[1].lower()]
            word_3_vec = vector[Qwords[2].lower()]
            x_vector = word_2_vec - word_1_vec + word_3_vec
            x_vector = torch.tensor(x_vector)
            x_vector_l2_norm = torch.dist(x_vector, torch.zeros_like(x_vector), 2)
            best_similarity = -1.0
            best_idx = None
            for word in w2i:
                word_idx = w2i[word]
                wordvec = vector[word]
                similarity = torch.dot(x_vector, wordvec) / (x_vector_l2_norm * l2_norm[word])
                if similarity > best_similarity and word_idx != w2i[Qwords[0].lower()] and word_idx != w2i[Qwords[1].lower()] and word_idx != w2i[Qwords[2].lower()]:
                #if similarity > best_similarity # 문제에 나온 단어들도 포함시키고 싶으면 위 조건문을 주석처리하고 이 문장을 사용
                    best_similarity = similarity
                    best_idx = word_idx
            #print(best_similarity)
            if Qwords[3].lower() == i2w[best_idx].lower():
                g.write("%s %s : %s %s?, Correct !!! \n" % (Qwords[0], Qwords[1], Qwords[2], i2w[best_idx].capitalize()))
                total_correct += 1.0
            else:
                g.write("%s %s : %s %s?, Wrong !!! Answer is %s \n" % (Qwords[0], Qwords[1], Qwords[2], i2w[best_idx].capitalize(), Qwords[3]))
    g.write("Questions = %d, Correct Questions = %d, Hitting Rate = %.4f%% \n" % (total_question, total_correct, (total_correct/total_question) *100.0))
    end_time = time.time()
    time_elapsed = end_time - start_time
    g.write("Time Elapsed for Validaiton %02d:%02d:%02d\n" % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    
    f.close()
    g.close()
    print('Questions = %d, Correct Questions = %d, Hitting Rate = %.4f' % (total_question, total_correct, (total_correct/total_question) *100.0))
    print('Time Elapsed for Validaiton %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    
def subsampling(input_seq, target_seq, discard_prob, mode = "SG", thresh = 0.00001):
###############################  Output  #########################################
# subsampled : Subsampled sequence                                               #
##################################################################################
    subsampling_start_time = time.time()

    remove_list = []
    input_seq_ssp = []
    target_seq_ssp = []
    discard = 0
    count = 0
    
    if mode == "CBOW":
        for t_idx in range(len(target_seq)):
            prob = discard_prob[target_seq[t_idx]]
            f = np.random.rand()
            if f >= prob:
                input_seq_ssp.append(input_seq[t_idx])
                target_seq_ssp.append(target_seq[t_idx])
            else:
                discard += 1
                
    elif mode == "SG":
        for i_idx in range(len(input_seq)):
            #count +=1
            #print(count)
            prob = discard_prob[input_seq[i_idx]]
            f = np.random.rand()
            if f >= prob:
                input_seq_ssp.append(input_seq[i_idx])
                target_seq_ssp.append(target_seq[i_idx])
            else:
                discard += 1

    subsampling_end_time = time.time()
    time_elapsed = subsampling_end_time - subsampling_start_time
    print('Time Elapsed for SubSampling %02d:%02d:%02d\n' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    print('Training Set before Subsampling :', len(input_seq))                
    print('Number of discard :', discard) 
    print('Training Set after Subsampling :', len(input_seq_ssp))

    return input_seq_ssp, target_seq_ssp 

def skipgram_HS(centerWord, contextCode, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# contextCode : Code of a contextword (type:str)                                  #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################
    
    #print(contextCode) #1010101111...
    #print(inputMatrix.shape) #3971 64
    #print(outputMatrix.shape) #K 64
    
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]
    K = outputMatrix.shape[0]

    hidden_layer = inputMatrix[centerWord].view(1,D)
    output_path = torch.mm(hidden_layer, outputMatrix.t()) # [1 D] X [D K] = [1 K]
    
    out = 1.0
    
    for path_node in range(K):
        if contextCode[path_node] == '0':
            #print('yes')
            out *= torch.sigmoid(output_path[0][path_node])
        else:
            #print('no')
            out *= torch.sigmoid(-output_path[0][path_node])
    
    loss = -torch.log(out)
    #print(loss)
    
    grad_node = torch.sigmoid(output_path) #[1 K]
    
    for path_node in range(K):
        if contextCode[path_node] == '0':
            grad_node[0][path_node] -= 1.0
    
    grad_out = torch.mm(grad_node.t(), hidden_layer)  #[K 1] X [1 D] = [K D]
    grad_in = torch.mm(grad_node, outputMatrix) #[1 K] X [K D] = [1 D]
    
    return loss, grad_in, grad_out


def skipgram_NS(centerWord, inputMatrix, outputMatrix):
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
    K = outputMatrix.shape[0]

    hidden_layer = inputMatrix[centerWord].view(1,D)
    output_layer = torch.mm(hidden_layer, outputMatrix.t()) # [1 D] X [D K] = [1 K]
    
    '''
    expout = torch.exp(output_layer)
    softmax = expout / expout.sum()

    loss = -torch.log(softmax[0][0])
    '''
    
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


def CBOW_HS(contextWords, centerCode, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# centerCode : Code of a centerword (type:str)                                    #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

        
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]
    C = len(contextWords)
    K = outputMatrix.shape[0]

    hidden_layer = inputMatrix[contextWords].sum(0).view(1,D) / C
    output_path = torch.mm(hidden_layer, outputMatrix.t()) # [1 D] X [D K] = [1 K]
    
    out = 1.0
    
    for path_node in range(K):
        if centerCode[path_node] == '0':
            #print('yes')
            out *= torch.sigmoid(output_path[0][path_node])
        else:
            #print('no')
            out *= torch.sigmoid(-output_path[0][path_node])
    
    loss = -torch.log(out)
    #print(loss)
    
    grad_node = torch.sigmoid(output_path) #[1 K]
    
    for path_node in range(K):
        if centerCode[path_node] == '0':
            grad_node[0][path_node] -= 1.0
    
    grad_out = torch.mm(grad_node.t(), hidden_layer)  #[K 1] X [1 D] = [K D]
    grad_in = torch.mm(grad_node, outputMatrix) #[1 K] X [K D] = [1 D]
    
    return loss, grad_in, grad_out


def CBOW_NS(contextWords, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################
    
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]
    C = len(contextWords)
    K = outputMatrix.shape[0]

    hidden_layer = inputMatrix[contextWords].sum(0).view(1,D) / C
    output_layer = torch.mm(hidden_layer, outputMatrix.t()) # [1 D] X [D K] = [1 K]
    
    target = output_layer[0][0]
    NS_samples = output_layer[0][1:]
    
    loss = -torch.log(torch.sigmoid(target)) - torch.sum(torch.log(torch.sigmoid(-NS_samples)))
    #print(loss)
    
    grad_output_layer = torch.sigmoid(output_layer) #[1 K]
    
    grad_output_layer[0][0] -= 1.0 #[1 K]
    
    grad_out = torch.mm(grad_output_layer.t(), hidden_layer) #[K 1] X [1 D] = [K D]
    grad_in = torch.mm(grad_output_layer, outputMatrix) #[1 K] X [K D] = [1 D]
    
    return loss, grad_in, grad_out


def word2vec_trainer(input_seq, target_seq, numwords, codes, stats, frequency, use_subsampling = True, threshold = 0.00001, mode="CBOW", NS=20, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()
    
    discard_prob = {}
    for word_id in frequency:
        discard_prob[word_id] = max(0, 1 - math.sqrt(threshold / frequency[word_id]))
    
    #sorted_codes = sorted(codes.items(), key=lambda kv: len(kv[1]))
    #print(len(sorted_codes[-1][1]))
    #print(target_seq)
    
    #각 node에 번호를 붙여준다.
    node = {}
    node[''] = 0
    node_index = 1
    for x in codes:
        #print(x, codes[x])
        for idx in range(1, len(codes[x])):
            if codes[x][:idx] in node:
                pass
            else:
                node[codes[x][:idx]] = node_index
                node_index += 1
    ''' 
    for x in node:
        print(x, node[x])
    '''

    #stats = torch.LongTensor(stats)

    for _ in range(epoch):
        #Training word2vec using SGD(Batch size : 1)
        
        if use_subsampling:
            input_seq_ssp, target_seq_ssp = subsampling(input_seq, target_seq, discard_prob, mode = mode, thresh = 0.00001)
        else:
            input_seq_ssp, target_seq_ssp = input_seq, target_seq

        for inputs, output in zip(input_seq_ssp,target_seq_ssp):
            i+=1
            #print(output)
            
            if mode=="CBOW":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    path = []
                    code = codes[output]
                    for idx in range(len(code)):
                        node_num = node[code[:idx]]
                        path.append(node_num)
                        
                    #activated = torch.ByteTensor(path)
                    activated = path
                    L, G_in, G_out = CBOW_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    #random.shuffle(stats)
                    NS_selected = random.sample(stats, NS)
                    while output in NS_selected:
                        NS_selected.remove(output)
                    activated = [output] + NS_selected
                    L, G_in, G_out = CBOW_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out

            elif mode=="SG":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    path = []
                    code = codes[output]
                    
                    for idx in range(len(code)):
                        node_num = node[code[:idx]]
                        path.append(node_num)
                        
                    #activated = torch.ByteTensor(path)
                    activated = path
                    #print(activated)
                    L, G_in, G_out = skipgram_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    #random.shuffle(stats)
                    NS_selected = random.sample(stats, NS)
                    while output in NS_selected:
                        NS_selected.remove(output)
                    activated = [output] + NS_selected
                    L, G_in, G_out = skipgram_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out

                
            else:
                print("Unkwnown mode : "+mode)
                exit()
            losses.append(L.item())
            #print(i)
            if i%50000==0:
            	avg_loss=sum(losses)/len(losses)
            	print("Loss : %f" %(avg_loss,))
            	losses=[]

    return W_in, W_out


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    mode = args.mode
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

    total_freq = 0
    #Code dict for hierarchical softmax
    freqdict={}
    freqdict[0]=10
    for word in vocab:
        freqdict[w2i[word]]=stats[word]
        total_freq += freqdict[w2i[word]]
    codedict = HuffmanCoding().build(freqdict)
    #print(freqdict)
    
    frequency = {} #for subsampling
    for word_id in freqdict:
        frequency[word_id] = freqdict[word_id] / total_freq
    
    #print(frequency)
    '''
    discard_prob = {}
    for word_id in frequency:
        discard_prob[word_id] = max(0, 1 - math.sqrt(thresh / frequency[word_id]))
    '''
    
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
    if mode=="CBOW":
        for j in range(len(words)):
            if j<window_size:
                input_set.append([0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)])
                target_set.append(w2i[words[j]])
            elif j>=len(words)-window_size:
                input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)])
                target_set.append(w2i[words[j]])
            else:
                input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)])
                target_set.append(w2i[words[j]])
 
    if mode=="SG":
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
    #print(len(input_set))
    #print(len(target_set))

    print("Vocabulary size")
    print(len(w2i))
    print()

    #Training section
    start_time = time.time()
    emb,_ = word2vec_trainer(input_set, target_set, len(w2i), codedict, freqtable, frequency, use_subsampling = False, threshold= 0.001, mode=mode, NS=ns, dimension=100, epoch=1, learning_rate=0.01)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Time Elapsed for Training %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    #embnp = emb.detach().numpy()
    #np.savez('./embedding.npz', emb = embnp)
    #embnp = torch.from_numpy(embnp)
    #emb = np.load('./embedding.npz')['emb']
    Analogical_Reasoning_Task(emb, w2i, i2w)

main()