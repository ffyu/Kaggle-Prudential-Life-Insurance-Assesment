# Kaggle-Prudential-Life-Insurance-Assesment

https://www.kaggle.com/c/prudential-life-insurance-assessment

My solution and ideas to start and finish a kaggle competition in a week.

**First, the solution.** 

To run **Final_Submission.py**, please first download two pre-processed data files - **train_processed.csv** and **test_processed.csv**, and put them in the same folder. Then change the variable **FOLDER** in the **Final_Submission.py** to the directory that you've downloaded the data files. This script will generate a file **submission.csv** as the final prediction results.

The two processed data files are produced using the following steps:

1) count the numbers of null variables row-wise

2) generate two additional features seperating the character and number of the feature "Product_Info_2"

3) add interaction term between BMI and Age

4) add counts of medical keywords

5) one-hot-encoding for 'Product_Info_2', 'Product_Info_2_char', 'Product_Info_2_num'

6) encode all the categorical features with the out-of-fold mean of the target values

The final submission is composed of two seperate models with slightly different paramters. The model weights are 2 and 1 for two models respectively. One extra step is added after the model training results to find the optimal splits to map the raw regression results to the oridinal risk measures.

**Second, the ideas to complete a competition in a short time (coming soon).**


