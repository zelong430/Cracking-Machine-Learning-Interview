# Cracking-Machine-Learning-Interview

The purpose of this repo is two fold:

* To help meslef and you prepare for data scienentist/ML engineer/DL engineer related interviews
* To summarize machine learning related knowledge in the form of easy to review Q&A

Most of the content are summarized from online resources including books, lectures, lecture notes, past exam. I would love to continously maintain and update this repo. Most of the answers are come from the original questions but some of them are added by myself. So consume with your own judgement and any correction/suggestions are welcome.

Any pull request are welcome.

* [Statistics and ML In General](#statistics-and-ml-in-general)
* [Classic Machine Learning](#classic-machine-learning)
    - [Supervised Learning](#supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
* [Reinforcement Learning](#reinforcement-learning)
* [Deep Learning](#deep-learning)
* [Natural Language Processing](#natural-language-processing)
* [Computer Vision](#computer-vision)

## Classic Machine Learning
### Supervised Learning

* [Linear Regression](#linear-regression)
* [Logistic Regression](#logistic-regression)
* [Naive Bayes](#naive-bayes)
* [KNN](#knn)
* [SVM](#svm)
* [Decision tree](#decision-tree)
* [Random forest](#random-forest)
* [Boosting Tree](#boosting-tree)
* [MLP](#mlp)
* [CNN](#cnn)
* [RNN and LSTM](#rnn-and-lstm)

#### Linear Regression
<img src="https://latex.codecogs.com/svg.latex?\Large&space;h_w{x}=w_0+w_1x_1+w_2x_2" />
 Or in vector form, 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;h_w{x}=w^Tx" />

* What is the cost function for linear regression?
    * Mean Squared Error
    * <img src="https://latex.codecogs.com/svg.latex?\Large&space;C=\frac{1}{2}\sum_{i=1}^{m}(h_w(x^{i}) - y^i)^2"/>

* What is the Normal Equation of linear regression?

* Suppose Pearson correlation between V1 and V2 is zero. In such case, is it right to conclude that V1 and V2 do not have any relation between them?
    * No, Pearson correlation coefficient between 2 variables might be zero even when they have a relationship between them. If the correlation coefficient is zero, it just means that that they don’t move together. We can take examples like y=|x| or y=x^2

* Suppose you have fitted a complex regression model on a dataset. Now, you are using Ridge regression with penality x. Is the bias going to be high or low in this case?
    * If the penalty is very large it means model is less complex, therefore the bias would be high.

* What will happen when you apply very large penalty in regularization?
    * In lasso some of the coefficient value become zero and the resulting parameter w would be sparse but in case of Ridge, the coefficients become close to zero but not zero.

* In which case, shall we use Lasso or Ridge regularization?
    * L1 (Lasso): can shrink certain coef to zero, thus performing feature selection
    * L2 (Ridge): shrink all coef with the same proportion; almost always outperforms L1
    * Elastic Net: combined L1 and L2 priors as regularizer

* Is linear regression sensetive to outliers in the dataset?
    * Yes, the slope of the regression line will change due to outliers in most of the cases. So Linear Regression is sensitive to outliers.





