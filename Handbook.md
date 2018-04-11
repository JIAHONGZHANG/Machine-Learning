- Method to predict a numeric output from statistics and machine learning:

  - Linear regression (statistics) determining the "line of best fit" using the least squares criterion(最小二乘法)
  - Linear models (machine learning) a predictive model from data under the assumption of a linear relationship between predictor and target variables

- Linear models, i.e outcome is linear combination of attributes
  $$
  y = b_0+b_1x_1+b_2x_2+...+b_nx_n
  $$

- Predicted value for first training instance $$X_1$$ is:
  $$
  b_0x_0^{(1)} + b_1x_1^{(1)} + b_2x_2^{(1)} + ...+b_nx_n^{(1)} =\sum_{i=0}^n{b_ix_i}
  $$

- Probability versus Statistics

  - Probability: reason from populations to sample
  - Statistics: reasons from sample to populations

- Expectation(期望) 
  $$
  E(X) = \sum_i{x_ip(X=x_i)}
  $$

- The **Sample** of **standard deviation** is 
  $$
  s = \sqrt{{1\over{N-1}}{\sum}_i(x_i-m)^2}
  $$

-  Variance(方差)
  $$
  \begin{equation}
  \begin{aligned}
  Var(X) &= E(X-E(X))^2  \\
  &=E(X^2)-[E(X)]^2
  \end{aligned}
  \end{equation}
  $$
  ​

- MSE and RMSE

  estimated value V

  true value $$\theta$$
  $$
  \begin{aligned}
  &MSE = {{\sum_{i=1}^n{(V_i - \theta_i)^2}}\over{n}}\\
  &RMSE = \sqrt{MSE}
  \end{aligned}
  $$

- Also 
  $$
  MSE = (variance) + (bias)^2
  $$
  the lowest possible value of MSE is 0

- Correlation
  $$
  \begin{aligned}
  &r = {{cov(x,y)}\over{{\sqrt{var(x)}}{\sqrt{var(y)}}}} \\
  &cov(x,y) = {{\sum}_i(x_i-\bar{x})(y_i-\bar{y})\over{n-1}}
  \end{aligned}
  $$
  Should be able to show that
  $$
  r = {{{\sum}_i(x_i-\bar{x})(y_i-\bar{y})}\over{\sqrt{\sum{(x_i-\bar{x})^2}}\sqrt{\sum{(y_i-\bar{y})^2}}}}
  $$
  ​

  - Case 1:  $$x_i>\bar{x}, y_i>\bar{y}$$
  - Case 2:  $$x_i<\bar{x}, y_i<\bar{y}$$
  - Case 3:  $$x_i<\bar{x}, y_i>\bar{y}$$
  - Case 4:  $$x_i>\bar{x}, y_i<\bar{y}$$

  In the first two case, $$x_i$$and $$y_i$$ vary together, both being high or low relative to their means.

  |r| in 0 to 1

- Suppose we want to investigate the relationship between people's height and weight. We collect n height and weight measurements.

  $$(h_i,w_i),1\leq i \leq n$$

  Univariate linear regression assumes a linear equation $$w=a+bh$$, with parameters a and b chosen such that the sum of **squared residuals**

  $$\sum_{i=1}^n{(w_i-(a+bh_i)^2)}$$ is **minimised**.

- Minkowski distance If$$\chi=\R ^d$$, the Minkowski distance of order $$p>0$$ is defined as
  $$
  Dis_p(X,Y)=(\sum_{j-1}^d|x_j-y_j|^p)^{1/p}=||X-Y||_p
  $$
  where $$||Z||_p=(\sum_{j=1}^d{|Z_j|}^p)^{1/p}$$ is the p-norm (some denoted $$L_p$$ norm) of the vector Z.

- The 2-norm refers to the familiar Euclidean distance
  $$
  Dis_2(X,Y)=\sqrt{\sum_{j=1}^d(x_j-y_j)^2} = \sqrt{(X-Y)^T(X-Y)}
  $$

- The 1-norm denotes Manhattan distance, also called cityblock distance
  $$
  Dis_1(X,Y) = \sum_{j=1}^d{|x_j-y_j|}
  $$
  ​

- Product Rule
  $$
  P(A\wedge B) = P(A|B)P(B) = P(B|A)P(A)
  $$

- Sum Rule
  $$
  P(A\vee B)=P(A)+P(B)-P(A\wedge B)
  $$

- Generally want the most probable hypothesis given the trainning data

  Maximum a posteriori hypothesis $$h_{MAP}$$:
  $$
  \begin{aligned}
  h_{MAP}&=arg\space max\space P(h|D) \\
  &=arg\space max\space {P(D|h)P(h)\over{P(D)}} \\
  &=arg\space max\space P(D|h)P(h)
  \end{aligned}
  \\
  h\in H
  $$

- If assume $$P(h_i)=P(h_j)$$ then can further simplify, and choose the 

  **Maximum likelihood (ML) hypothesis**
  $$
  h_{ML} = arg\space max\space P(D|h_i)\\
  h_i\in H
  $$

- Example
  $$
  \begin{aligned}
  &P(cancer)=0.008\space &P(\urcorner cancer)=0.992 \\
  &P(\oplus|cancer)=0.98\space &P(\ominus|cancer)=0.02\\
  &P(\oplus|\urcorner cancer)=0.03\space &P(\ominus|\urcorner cancer)=0.97\\
  \end{aligned}
  $$
  maximum a posteriori (MAP)
  $$
  \begin{aligned}
  P(\oplus|cancer)P(cancer)&=0.98\times 0.008=0.00784\\
  P(\oplus|\urcorner cancer)P(\urcorner cancer)&=0.03\times 0.992=0.02976
  \end{aligned}
  $$
  Thus $$h_{MAP}=\urcorner cancer$$ 

- Multivariate Bernoulli model and multinomial model

  more detail look at slide chapter3 111

- Perceptron



- Entropy
  $$
  Entropy(S)\equiv -p_{\oplus}log_2P_{\oplus}-p_{\ominus}log_2p_{\ominus}
  $$

- Gain(S, A)

  expected reduction in entropy due to sort on A
  $$
  Gain(S,A)=Entropy(S)-\sum_{v\in Values(A)}{|S_v|\over|S|}Entropy(S_v)
  $$
  ​

