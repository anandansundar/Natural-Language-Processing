# Natural-Language-Processing
Projects done in Natural Language Processing at The University of Texas at Dallas

This project does text classification based on the wikipedia content.

Please make sure that we have a stable internet connection.
Put all the files which is extracted in the same directory. 

Running the baseline system:  

`python3.5 baseline.py`

Running the improved system:
We can run training_improved.py to train the system. But this takes approximately 30 to 40 minutes.  

`python3.5 training_improved.py`

it will generate a model file naivebayes_improved4.pickle. We have also attached the model with the submission.
To test the model against our test data set,   

`python3.5 testing_improved.py`

This will take the model from the folder and then classifies the test set.
