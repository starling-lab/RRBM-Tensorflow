# RRBM-Tensorflow
code for "Relational Restricted Boltzmann Machines- A Probabilistic Logic Learning Approach"

1.	In order to execute this code, the following input files are required:
  a.	DataSetName/LiftedRW_Schema.txt:
       This file contains the schema that is required to create the lifted random walks for the model. Flags like ‘NoBF’, ‘NoTwin’ etc.          required in the above file can be set by referring to [1].
  b.	DataSetName/5Folds/FoldNumber/Training/TrainingPosExample.db and DataSetName/5Folds/FoldNumber/Test/TestPosExample.db: these files 
      should contain the training/test positive examples.
  c.	Similarly, DataSetName/5Folds/FoldNumber/Training/TrainingNegExample.db and DataSetName/5Folds/FoldNumber/Test/TestNegExample.db 
      should contain training and test negative examples respectively.
  d.	DataSetName/5Folds/FoldNumber/Training/TrainingFacts.db should contain all the facts, including the inverted facts (one starting 
      with _ ) in it. Likewise, for TestFacts.db.
  e.	You should also create an empty folder named “graphs” (at the same level as DataSetName Folder) which will be used (internally) to 
      ground the lifted random walks.
  f.	Kindly create the above folders and files with the same nested structure as explained above.
  g.	The program will internally create rest of the files, RandomWalks.txt, RWRPredicates.txt, schema.db, countVecs.txt, DataSet.csv. 
2.	kindly set all the parameters of the model in line 13-24 of RRBM-Main.py
3.	Now you are ready to execute the main file – RRBM-Main.py. It does not require command line arguments as they have been set in step B.
Disclaimer: This code was modified after the acceptance of ILP paper [2]. Hence, the results produced by this code might vary slightly from the one presented in the published paper.

References:
1.	Ni Lao and William W. Cohen (2010), “Relational Retrieval Using a Combination of Path-Constrained Random Walks”, ECML.
2.	Navdeep Kaur, Gautam Kunapuli, Tushar Khot, Kristian Kersting, William Cohen and Sriraam Natarajan(2017), “Relational Restricted Boltzmann Machines: A Probabilistic Logic Learning Approach”, ILP.
