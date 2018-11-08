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
The formulation for linear regression: 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;h_w(x)=w_0+w_1x_1+w_2x_2" />

Or in vector form, 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;h_w(x^i)=w^Tx^i" />

* What is the cost function for linear regression?
    * Mean Squared Error
    * <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{2}\sum_{i=1}^{m}(h_w(x^i)-y^i)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{2}\sum_{i=1}^{m}(h_w(x^i)-y^i)^2" title="\frac{1}{2}\sum_{i=1}^{m}(h_w(x^i)-y^i)^2" /></a>

* What is the Normal Equation of linear regression?
    * Vector calculus:
        suppose X is a n * d matrix where each row corresponding to a data point and each colume corresponding to a feature. The cost function could be expressed as:
        <a href="https://www.codecogs.com/eqnedit.php?latex=||Xw&space;-&space;\overline{y}||_2^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?||Xw&space;-&space;\overline{y}||_2^2" title="||Xw - \overline{y}||_2^2" /></a>

        Therefore, the gradient of cost function with regarding to w is:

        <a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangledown_w&space;||Xw&space;-&space;\overline{y}||_2^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangledown_w&space;||Xw&space;-&space;\overline{y}||_2^2" title="\bigtriangledown_w ||Xw - \overline{y}||_2^2" /></a>

        =<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangledown_w&space;(Xw&space;-&space;\overline{y})^T&space;(Xw&space;-&space;\overline{y})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangledown_w&space;(Xw&space;-&space;\overline{y})^T&space;(Xw&space;-&space;\overline{y})" title="\bigtriangledown_w (Xw - \overline{y})^T (Xw - \overline{y})" /></a>

        =<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangledown_w&space;(w^TX^TXw-w^TX^Ty-y^TXw-y^Ty)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangledown_w&space;(w^TX^TXw-w^TX^Ty-y^TXw-y^Ty)" title="\bigtriangledown_w (w^TX^TXw-w^TX^Ty-y^TXw-y^Ty)" /></a>

        =<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangledown_ww^TX^TXw-2\bigtriangledown_wy^TXw" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangledown_ww^TX^TXw-2\bigtriangledown_wy^TXw" title="\bigtriangledown_ww^TX^TXw-2\bigtriangledown_wy^TXw" /></a>

        Using the results from matrix calculus:

        <a href="https://www.codecogs.com/eqnedit.php?latex=\\\bigtriangledown_xw^Tx=w&space;\newline&space;\bigtriangledown_xx^TAx=2A,&space;\text{where&space;A&space;is&space;symmetrix&space;matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\\bigtriangledown_xw^Tx=w&space;\newline&space;\bigtriangledown_xx^TAx=2A,&space;\text{where&space;A&space;is&space;symmetrix&space;matrix}" title="\\\bigtriangledown_xw^Tx=w \newline \bigtriangledown_xx^TAx=2Ax, \text{where A is symmetrix matrix}" /></a>

        <a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangledown_ww^TX^TXw-2\bigtriangledown_wy^TXw=2X^TXw-2X^Ty" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangledown_ww^TX^TXw-2\bigtriangledown_wy^TXw=2X^TXw-2X^Ty" title="\bigtriangledown_ww^TX^TXw-2\bigtriangledown_wy^TXw=2X^TXw-2X^Ty" /></a>

        By setting the gradient equal to 0, we have
        <a href="https://www.codecogs.com/eqnedit.php?latex=w&space;=&space;(X^TX)^{-1}X^Ty" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w&space;=&space;(X^TX)^{-1}X^Ty" title="w = (X^TX)^{-1}X^Ty" /></a>


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





