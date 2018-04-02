# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:22:52 2017

@author: Navdeep Kaur
"""
from call_java_files import obtain_lifted_random_walks, create_schema_file_for_counting, change_random_walks_format
from call_java_files import sample_random_walks, obtain_counts_of_random_walks, generated_normalized_data, read_input_data
import tensorflow as tf
import numpy as np
import subprocess

RWSchema_Path           = r"C:/Users/navdeep/Desktop/RRBM-tensorflow/imdb/LiftedRW_Schema.txt"
RW_Sample_Path          = r"C:/Users/navdeep/Desktop/RRBM-tensorflow/imdb/5Folds/Fold1/"
startentity             = "person"
endentity               = "person"
targetpredicate         = "workedUnder"
randomwalklength        = "6"
sample_random_walk_size = 100
DataSetName             = "imdb"
FoldNumber              = "Fold1"

learning_rate           = 0.05
training_epochs         = 2

#x = tf.placeholder("float", [sample_random_walk_size,1])
#Y = tf.placeholder("float", [2,1])


def init_crbm_weights(shape,name1):
        # initilize weigts from uniform distribution. As described in Learning Algorithms for the Classification Restricted Boltzmann machine
        m = shape[0]
        n = shape[1]
        M = max(m,n)
        interval_max = M**(-0.5)
        interval_min = -interval_max
        weights = tf.Variable(tf.random_uniform([m,n], minval = interval_min, maxval = interval_max, dtype=tf.float32), name = name1) 
   
        return weights

def init_crbm_bias(size,name):
    bias = tf.Variable(tf.constant(0.1, dtype = tf.float32, shape=size),name=name)
    return bias
    
def initialize_weights(visible_size,hidden_size,n_classes):
    
    
    W = init_crbm_weights([hidden_size,visible_size],"W")
    U = init_crbm_weights([hidden_size,n_classes], "U")
    
    b = init_crbm_bias([visible_size,1], "b")
    c = init_crbm_bias([hidden_size,1], "c")
    d = init_crbm_bias([n_classes,1], "d")
      
    return W, U, b, c, d

def softplus(O):
    return tf.log(1+tf.exp(O))

def restricted_boltzmann_machines_network(x, W, U, b, c, d, visible_size,hidden_size,n_classes):
    

    W_x         = tf.add(tf.matmul(W, tf.transpose(x)), c)
    
    U_jy        = tf.convert_to_tensor(tf.split(U, num_or_size_splits=n_classes, axis=1))
    
    Cj_Ujy_W_x  = tf.add(W_x, U_jy)
    
    O           = softplus(Cj_Ujy_W_x)
    
    sum_softmax = tf.reduce_sum(O,1)
    
    F           = tf.add(d, sum_softmax)
    
    F_t =      tf.transpose(F)
    return F_t
   
        
################## MAIN CODE STARTS HERE ######################################################################################    
obtain_lifted_random_walks(RWSchema_Path, startentity, endentity,randomwalklength)
change_random_walks_format(RWSchema_Path)
create_schema_file_for_counting(RWSchema_Path,startentity,endentity,targetpredicate)
sample_random_walks(RWSchema_Path, RW_Sample_Path, sample_random_walk_size)
obtain_counts_of_random_walks(DataSetName,FoldNumber)
generated_normalized_data(RW_Sample_Path)

pickle_training_path = RW_Sample_Path + "Training/DataSet.csv"
pickle_test_path     = RW_Sample_Path + "Test/DataSet.csv"
AUCpath =              RW_Sample_Path + "Test/AUCROC.txt";

training_data_X, training_data_Y, DataSetSize,     visible_size     , n_classes      = read_input_data(pickle_training_path) 
test_data_X,     test_data_Y,     DataSetSizeTest, visible_size_test, n_classes_test = read_input_data(pickle_test_path) 

x = tf.placeholder("float", [1,visible_size])
Y = tf.placeholder("float", [1,2])

hidden_size             = int(0.6*visible_size)

[W,U,b,c,d]             = initialize_weights(visible_size,hidden_size,n_classes)

tf.summary.histogram("W_summary",W)

logits                  =  restricted_boltzmann_machines_network(x, W, U, b, c, d, visible_size, hidden_size, n_classes)

#define the loss function
cost         = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer    = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

prediction   = tf.nn.softmax(logits) 
correct_pred = tf.equal(tf.argmax(Y,1), tf.argmax(prediction, 1))
acc_op       = tf.reduce_mean(tf.cast(correct_pred, "float"))

# Start with reading new input
init = tf.global_variables_initializer()

with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./RRBM", sess.graph) # for 0.8
        
        for epoch in range(training_epochs):
            total_batch = int(DataSetSize)
            for i in range(total_batch):
                #print(training_data_Y[i].shape)
                batch_x = np.reshape(training_data_X[i], [1,visible_size])
                batch_y = np.transpose(np.reshape(training_data_Y[i], [n_classes,1]))
                              
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, Y: batch_y})
                
                summary, _ = sess.run([merged, cost], feed_dict={x: batch_x, Y: batch_y})
                writer.add_summary(summary, i)

# TEST PHASE
        count = 0;
        test_accuracy = 0.
        AUCROC =[]
        total_batch_test = int(DataSetSizeTest)
        for j in range(total_batch_test):
            
            batch_x_test = np.reshape(test_data_X[j], [1,visible_size_test])
            batch_y_test = np.transpose(np.reshape(test_data_Y[j], [n_classes_test,1]))
			
            accur = sess.run(acc_op, feed_dict={x: batch_x_test, Y: batch_y_test})  
            test_accuracy += accur
            pred  = sess.run(prediction, feed_dict={x: batch_x_test, Y: batch_y_test})
            x_arg0 = "{:.9f}".format(pred[0][0])
            x_arg1 = "{:.9f}".format(pred[0][1])
            y_arg0 = int(batch_y_test[0][0])
            y_arg1 = int(batch_y_test[0][1])
            #z1 = str(x_arg0)+"\t"+str(x_arg1)+"\t"+ str(y_arg0)+" \t"+str(y_arg1)
            #print(z1+"**")
            z = str(x_arg1)+" \t"+str(y_arg1)
            AUCROC.append(z)
            del(batch_x_test)
            count = count +1
        
                    
        f= open(AUCpath,'w')
        for i1 in range(len(AUCROC)):
            f.write(AUCROC[i1]+"\n")
        f.close() 
                  
process = subprocess.Popen(['java', '-jar', 'auc.jar',AUCpath,'list'],stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
output, error = process.communicate()
lines = output.split(b'\n')
for s in lines:
    if b'Area Under the Curve for ROC' in s:
        c = str(s)
        print(c[2:-3])
    if b'Precision - Recall' in s:
        d = str(s)
        print(d[2:-3])       
