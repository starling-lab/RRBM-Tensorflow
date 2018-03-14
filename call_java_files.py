# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 18:03:06 2017

@author: nxk176630
"""
import subprocess
import random
import os

import numpy as np
from numpy import genfromtxt

def obtain_lifted_random_walks(trainpath, startentity, endentity,randomwalklength):
        return subprocess.call(['java', '-jar', 'RandomWalks.jar','-rw','-train', trainpath, '-startentity', str(startentity),'-endentity',str(endentity),'-maxRWlen',str(randomwalklength)])
def obtain_counts_of_random_walks(DataSetName,FoldNumber):
        DataSetFold = DataSetName+FoldNumber
        
        countPath = "./"+DataSetName+"/5Folds/"+FoldNumber+"/"
        
        if not os.path.exists("./graphs"):
            os.makedirs("./graphs")
        if os.path.exists(countPath+"Training/countVecs.txt"):
            os.remove(countPath+"Training/countVecs.txt")
        if os.path.exists(countPath+"Test/countVecs.txt"):
            os.remove(countPath+"Test/countVecs.txt")
        
        return subprocess.call(['java', '-jar', 'propCnt.jar', str(DataSetName),str(FoldNumber),str(DataSetFold)])


def write_to_file(directory, facts):
    if os.path.exists(directory):
        os.remove(directory)
    
    with open(directory, 'w') as pos_file:
        for f in facts:
            pos_file.write(f+"\n")

def read_my_file(filename):
    file_content = []
    with open(filename, 'r') as file:
        for line in file:
            if "\n" in line:
                file_content.append(line[:-1])
            else:
                file_content.append(line)
    file.close()
    return file_content

def create_schema_file_for_counting(trainpath,startentity,endentity,targetpredicate):
    
    m       = trainpath.split('/')
    dirname = "/".join(m[:-1])
    
    inv_pred_list =[]
    filename = dirname+"/schema.db"
    
    with open(trainpath, 'r') as file:
        for line in file:
            fact = line.split("|")
                        
            if(len(fact)==1):
                 m=fact[0][:-1].split(",")
                 inv_pred_list.append(fact[0][:-1])
            else:
                 if "NoTwin" in fact[1]:
                     inv_pred_list.append(fact[0]) 
                     continue
                 else:
                    inv_pred_list.append(fact[0]) 
                    m = fact[0].split(",")
            n = m[0].split("(")
            inv_pred = "_"+n[0]+"("+m[1][:-1]+","+n[1]+")"
            inv_pred_list.append(inv_pred)
        target = targetpredicate+"("+startentity+","+endentity+")"
        inv_pred_list.append(target)
    
    write_to_file(filename, inv_pred_list)
    
def change_random_walks_format(trainpath):
    m       = trainpath.split('/')
    dirname = "/".join(m[:-1])
    
    rwpath = dirname+"/RWRPredicates.txt"
    outputfile = dirname+"/RandomWalks.txt"
    rwstring=""
    rw_list=[]
    
    with open(rwpath, 'r') as file:
        for line in file:
            count = 0
            fact = line.split("),")
            for f in fact:
                arg1 = f.split(",")
                pred = arg1[0].split("(")
                a0 = pred[1][0]+str(count)
                count = count + 1
                a1 = arg1[1][0]+str(count)
                
                predicate = pred[0]+"("+a0+","+a1+")"
                
                rwstring = rwstring+predicate+"^"
            rw_list.append(rwstring[:-1])
           # print(line,rwstring[:-1],"*")
            rwstring=""
    
    write_to_file(outputfile, rw_list)         
            
def sample_random_walks(RWSchema_Path, RW_Sample_Path, sample_random_walk_size):
    
    m       = RWSchema_Path.split('/')
    dirname = "/".join(m[:-1])
    
    filename = dirname+"/RandomWalks.txt"
    
    All_Random_Walks=read_my_file(filename) 
    sampled_walks = random.sample(All_Random_Walks, sample_random_walk_size)
    
    training_path = RW_Sample_Path+"Training/RandomWalks.txt"
    test_path     = RW_Sample_Path+"Test/RandomWalks.txt"
    
    write_to_file(training_path, sampled_walks)
    write_to_file(test_path, sampled_walks)

def get_max_value_from_file(RW_Sample_Path):
    training_path = RW_Sample_Path+"Training/countVecs.txt"
    test_path     = RW_Sample_Path+"Test/countVecs.txt"
    
    with open(training_path, 'r') as file:
        maxvaluetrain = 1
        for line in file:
             fact = line.split(",")
             for f in fact:
                 if maxvaluetrain < int(f):
                     maxvaluetrain = int(f)
        
    with open(test_path, 'r') as file:
        maxvaluetest = -2
        for line in file:
             fact = line.split(",")
             for f in fact:
                 if maxvaluetest < int(f):
                     maxvaluetest = int(f)
    
    
    if maxvaluetrain > maxvaluetest:
        returnedvalue = 1.5 * maxvaluetrain
    else:
        returnedvalue = 1.5 * maxvaluetest
    
    return returnedvalue

def normalized(training_path, MaxValue):
    rw_listtrain=[]
    count = 0
    mylen = 0
    #print(MaxValue)
    with open(training_path, 'r') as file:
        for line in file:
            line = line.rstrip()
            mynewline = ""
            fact = line.split(",")
            mylen = len(fact)
            for f in fact:
                if count is mylen-1:
                    mynewline = mynewline+fact[-1]
                    count = 0
                else:
                    f1 = float(f)/MaxValue
                    mynewline = mynewline+repr(f1)+","
                    count = count + 1
           # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
          #  print(line+"*")
           # print(mynewline+"$")
            rw_listtrain.append(mynewline)   
    return rw_listtrain
    
    
def generated_normalized_data(RW_Sample_Path):
    
    RWSchema_Train = RW_Sample_Path+"Training/countVecs.txt"
    RWSchema_Test  = RW_Sample_Path+"Test/countVecs.txt"
    
    trainfile      = RW_Sample_Path+"Training/DataSet.csv"
    testfile       = RW_Sample_Path+"Test/DataSet.csv"
    
    MaxValue      = get_max_value_from_file(RW_Sample_Path)
    train_list    = normalized(RWSchema_Train, MaxValue)
  #  print(len(train_list))
    test_list     = normalized(RWSchema_Test, MaxValue)
    
         
    write_to_file(trainfile, train_list)
    write_to_file(testfile,  test_list)
    
def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


def read_input_data(RW_Sample_Path):
    my_data       = genfromtxt(RW_Sample_Path, delimiter=',')
    np.random.shuffle(my_data)
    mydatasize    = my_data.shape[0]
    features      = my_data[:,0:-1]
    myfeaturesize = features.shape[1]
    
    target        = convertToOneHot((my_data[:,-1]).astype(int))
    mytargetsize  = target.shape[1]
    
    features1 = features.astype(float)
    target1   = target.astype(float)
    
    
    return features1, target1, mydatasize, myfeaturesize, mytargetsize
    

#RW_Sample_Path = 'C:/Users/nxk176630/Desktop/RRBM-tensorflow/imdb/5Folds/Fold1/Training/DataSet.csv'
#read_input_data(RW_Sample_Path)


