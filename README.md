# UE17CS303 - MACHINE LEARNING ASSIGNMENT
##### Applying Naive Bayes classifier to predict voting pattern in a two party electoral system.
![Machine learning](./images/machine-learning.png)

## INTRODUCTION
The objective of this project is to apply Naive Bayes classifier on 1984 American election dataset to predict whether a voter is republican or democrat. The dataset has 435 entries each consisting of 16 binary attributes and one target attribute which is also binary in nature. So basically, the dataset has 16⨯435 data points out of which 392 are missing. Since all the attributes are binary, we have converted all ‘yes’ and ‘no’ to ones and zeros. We have also assumed democrat as positive classification and republican as negative.

## DATA CLEANING AND VISUALIZATION
As we have discussed before, our house-votes dataset has 16⨯435 data points out of which 392 are missing and these missing values are distributed over 203 rows and last attribute has about 25% data missing, hence we can’t perform list wise deletion.
![Missing data-points in the dataset](./images/missing-datapoints-in-the-dataset.png)
Yellow lines in the above figure show the distribution of missing values in all attributes.

So, what we have done is as follows:

We have plotted the distribution of ‘yes’ and ‘no’ for all the 16 attributes across the target attributes: democrat and republican. And now we have used this distribution to impute the missing values.

Suppose the `duty free import` attribute for the second row is missing and the target attribute for that row is democrat. Then we see the distribution of ‘yes’ and ‘no’ in the `duty free import` attribute for all the voters who are democratic. Now whichever attribute value has higher count, we replace the missing value with that attribute value.

The graphs below show the above mentioned distributions:

<img src="./images/adoption-of-the-budget-resolution.png" width="50%" height="50%"/>
<img src="./images/aid-to-nicaraguan-contras.png" width="50%" height="50%"/>
<img src="./images/anti-satellite-test-ban.png" width="50%" height="50%"/>
<img src="./images/crime.png" width="50%" height="50%"/>
<img src="./images/duty-free-exports.png" width="50%" height="50%"/>
<img src="./images/education-spending.png" width="50%" height="50%"/>
<img src="./images/el-salvador-aid.png" width="50%" height="50%"/>
<img src="./images/export-administration-act-south-africa.png" width="50%" height="50%"/>
<img src="./images/handicapped-infants.png" width="50%" height="50%"/>
<img src="./images/immigration.png" width="50%" height="50%"/>
<img src="./images/mx-missile.png" width="50%" height="50%"/>
<img src="./images/religious-groups-in-schools.png" width="50%" height="50%"/>
<img src="./images/superfund-right-to-sue.png" width="50%" height="50%"/>
<img src="./images/synfuels-corporation-cutback.png" width="50%" height="50%"/>
<img src="./images/water-project-cost-sharing.png" width="50%" height="50%"/>

## IMPLEMENTATION OF NAÏVE BAYES CLASSIFIER AND 5-FOLD CROSS VALIDATION
For implementing 5-fold cross validation, we have divided the dataset into 5 bins and in each of the 5 iterations, one bin is used as test-set and remaining four are used as train-set.
Naïve Bayes Classifier is a probabilistic classifier based on Bayes theorem which uses both conditional probability and simple probability and is given by:

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a})

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?\Large&space;h_{MAP})


