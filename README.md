# Kaggle-Prudential-Life-Insurance-Assesment

https://www.kaggle.com/c/prudential-life-insurance-assessment

**Final private leader board ranking: 48/2619 (top 2%)**

My solution and ideas to start and finish a kaggle competition in a week.

**First, the solution.** 

To run **Final_Submission.py**, please first download two pre-processed data files - **train_processed.csv** and **test_processed.csv**, and put them in the same folder. Then change the variable **FOLDER** in the **Final_Submission.py** to the directory that you've downloaded the data files. This script will generate a file **submission.csv** as the final prediction results.

The two processed data files are produced using the following steps:

1) count the number of null feature values row-wise

2) generate two additional features seperating the character and number parts of the feature "Product_Info_2"

3) add interaction term between BMI and Age

4) add counts of medical keywords

5) one-hot-encoding for 'Product_Info_2', 'Product_Info_2_char', 'Product_Info_2_num'

6) encode all the categorical features with the out-of-fold mean of the target

The final submission is composed of two seperate models with slightly different paramters. The model weights are 2 and 1 respectively. One extra step is added after the model training to find the optimal splits. This step maximizes the quadratic weighted kappa with respect to 7 cut-off points. With those optimal cut-off points, the final prediction is then caculated by mapping the raw regression results to the oridinal risk measures.

**Second, the ideas to complete a competition in a short time (coming soon).**

I started this competition a week before the deadline and had about 5 weekday nights plus the weekends to figure things out. I'll list the tasks for each day and hopefully that would provide a framework to manage the time and resources if a short turnaround is required.


