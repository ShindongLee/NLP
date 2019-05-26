import torch
import csv
import argparse
import numpy as np
import time
import math
import random

from preprocess import read_data

torch.set_printoptions(threshold=10000)

def validation(w_in, w_out, w2i, use_bigram):
    start_time = time.time()
    test_idx = 0
    
    count = [0, 0, 0, 0]
    correct = [0, 0, 0, 0]
    
    article_class = {0:'World', 1:'Sports', 2:'Business', 3:'Sci/Tech'}
    g = open("result.txt", 'w')

    with open('./ag_news_csv/test.csv') as train_csv:
        fullcsv = csv.reader(train_csv, delimiter = ',')
        for line in fullcsv:
            test_idx += 1
            answer = int(line[0]) - 1
            count[answer] += 1
            title = line[1]
            body = line[2]
            title_body = title + ' ' + body
            word_list = title_body.split()
            word_idx_lst = []
            
            if use_bigram == 'False':
                for word in word_list:
                    if word in w2i:
                        word_idx_lst.append(w2i[word])

            elif use_bigram == 'True':
                for word_idx in range(len(word_list) - 1):
                    bigram = word_list[word_idx] + '-' + word_list[word_idx + 1]
                    if bigram in w2i:
                        word_idx_lst.append(w2i[bigram])
            
            hidden = w_in[word_idx_lst].sum(0).view(1,-1) / len(word_idx_lst)
            out = torch.mm(hidden, w_out.t())
            e = torch.exp(out)
            softmax = e / torch.sum(e, dim=1, keepdim = True)
            classification = torch.argmax(softmax).item()
            
            if classification == answer:
                correct[classification]+=1
                print('Correct !!! {:<5} th article is classified as {:<10} ,actually is  {:<10}'.format(test_idx, article_class[classification], article_class[answer]))
                g.write("Correct !!! {:<5} th article is classified as {:<10} ,actually is  {:<10}\n".format(test_idx, article_class[classification], article_class[answer]))
            else:
                print('Wrong   !!! {:<5} th article is classified as {:<10} ,actually is  {:<10}'.format(test_idx, article_class[classification], article_class[answer]))
                g.write("Wrong   !!! {:<5} th article is classified as {:<10} ,actually is  {:<10}\n".format(test_idx, article_class[classification], article_class[answer]))
        
        print('\n')
        print('Number of Questions: %d, Number of Correct Questions: %d' %(test_idx, sum(correct)))
        print('Total Hit Rate is %.5f %%' %((sum(correct) / test_idx) * 100.0))

        g.write("\n")
        g.write("Number of Questions: %d, Number of Correct Questions: %d\n" %(test_idx, sum(correct)))
        g.write("Total Hit Rate is %.5f %%\n\n" %((sum(correct) / test_idx) * 100.0))
        
        for c in range(4):
            print('Hit Rate of Class {:<10}: {:.5f} %'.format(article_class[c], ((correct[c] / count[c])*100)))
            g.write("Hit Rate of Class {:<10}: {:.5f} %\n".format(article_class[c], ((correct[c] / count[c])*100)))
            
    print('\n')
    g.write("\n")
    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Time Elapsed for Validaiton %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    g.write("Time Elapsed for Validaiton %02d:%02d:%02d\n" % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))


def train(word_idx_lst, answer, inputMatrix, outputMatrix):
################################  Input  ##########################################
# word_idx_lst : Indices of contextwords (type:list(int))                         #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(4,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(4,D))                    #
###################################################################################
    
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]
    C = len(word_idx_lst)
    K = outputMatrix.shape[0]

    hidden_layer = inputMatrix[word_idx_lst].sum(0).view(1,D) / C
    output_layer = torch.mm(hidden_layer, outputMatrix.t()) # [1 D] X [D 4] = [1 4]

    e = torch.exp(output_layer)
    softmax = e / torch.sum(e, dim=1, keepdim = True)
    
    loss = -torch.log(softmax[0][answer])
    
    grad_output_layer = softmax #[1 4]
    
    grad_output_layer[0][answer] -= 1.0 #[1 4]
    
    grad_out = torch.mm(grad_output_layer.t(), hidden_layer) #[4 1] X [1 D] = [4 D]
    grad_in = torch.mm(grad_output_layer, outputMatrix) #[1 4] X [4 D] = [1 D]
    
    return loss, grad_in, grad_out


def word2vec_trainer(train_data, labels, numwords, dimension=10, learning_rate=0.001, epoch=10):

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    W_out = torch.randn(4, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples")
    print(len(train_data))

    for _ in range(epoch):
        
        #shuffle train data every epoch
        match = list(zip(train_data, labels))
        random.shuffle(match)
        train_data, labels = zip(*match)
    
        #Training word2vec using SGD(Batch size : 1)
        for word_idx_lst, answer in zip(train_data, labels):
            i+=1
            #print(output)
            L, G_in, G_out = train(word_idx_lst, answer, W_in, W_out)
            W_in[word_idx_lst] -= learning_rate*G_in
            W_out -= learning_rate*G_out
            
            losses.append(L.item())
            #print(i)
            if i%5000==0:
            	avg_loss=sum(losses)/len(losses)
            	print("Loss : %f" %(avg_loss,))
            	losses=[]

    return W_in, W_out


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('ub', metavar='use_bigram', type=str,
                        help='True for using bigram, False for not using bigram')
    parser.add_argument('d', metavar='dimension', type=int,
                        help='Dimension of hidden layer')
    parser.add_argument('e', metavar='epoch', type=int,
                        help='Total epoch for training')
    parser.add_argument('lr', metavar='learning_rate', type=float,
                        help='Learning Rate for training')
    
    args = parser.parse_args()
    
    use_bigram = args.ub
    dimension = args.d
    epoch = args.e
    learning_rate = args.lr
    
    assert use_bigram == 'True' or use_bigram == 'False'
    
    print("preprocessing...")
    print('Using Bigram:',use_bigram)
    
    train_data, labels, w2i = read_data(use_bigram)

    #Training section
    start_time = time.time()
    w_in, w_out = word2vec_trainer(train_data, labels, len(w2i), dimension = dimension, epoch = epoch, learning_rate=learning_rate)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Time Elapsed for Training %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    validation(w_in, w_out, w2i, use_bigram)

main()