---
layout: page
title: Tutorial
---

# Tutorial

<br>

<p align="center"><img src="/assets/tutorial/logo.png" width=550 /></p>


## Introduction
Modern methods in machine learning have brought about unprecedented predictive performance in a wide range of tasks and applications. While the accuracy of predictions has improved, the need to reason about the uncertainty that accompany these predictions has also grown. Awareness of uncertainty is especially important in application domains like healthcare, criminal justice, and autonomous systems, where some degree of confidence in a prediction is needed in order to make better informed decisions. Estimating this uncertainty is the key problem of predictive uncertainty quantification (UQ).

Despite the machine learning community’s increased interest in UQ, there is a lack of good libraries for predictive uncertainty evaluation. Standard libraries, such as scikit-learn, provide some basic functions but do not have implementations for more recent evaluation metrics. While some implementations of these metrics exist in scattered sources online, there is no central library that comprises all the tools needed to do evaluation. 

To address this, we present [Uncertainty Toolbox](https://github.com/uncertainty-toolbox/uncertainty-toolbox): a comprehensive library containing evaluation metrics, visualizations, and recalibration tools. The toolbox has a simple interface, making it easy to visualize predictive uncertainties or compute many different metrics at once. Alongside this suite of tools, we provide a description of all metrics, a glossary of key terms, and a list of important papers for those unfamiliar with the field. All of this can be found in our GitHub repository: [https://github.com/uncertainty-toolbox/uncertainty-toolbox](https://github.com/uncertainty-toolbox/uncertainty-toolbox)

In this blog post, we will:
Provide a brief background on UQ by describing the key concepts and evaluation metrics for those unfamiliar with the topic.
Show how to easily compute metrics for, visualize, and recalibrate predictive uncertainties by walking through a use case of Uncertainty Toolbox in a [Colab notebook](https://colab.research.google.com/drive/1fg61MFmVmgFYM4CRm_aoiZ_WRdxbIhIe?usp=sharing).


## Key Concepts in Predictive Uncertainty

In this section we describe some key concepts in predictive uncertainty for those unfamiliar with the field. Readers that are familiar with these concepts are free to skip to the next section where we demonstrate exactly how to use the toolbox.

Note: see the [glossary page](https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/master/docs/glossary.md) of the toolbox for more thorough coverage and explanations of these concepts.

Predictive uncertainties can be expressed with distributional predictions. For example, in the classification setting, predicting a multinomial distribution is a common method of expressing uncertainty over class labels. In the regression setting, a widely used method is to predict a mean and a standard deviation to parametrize the output distribution with a conditional Gaussian distribution. Regardless of the form, predictive uncertainty can be evaluated in terms of calibration and sharpness.
Calibration refers to the degree to which a predicted distribution matches the true underlying distribution of the data. For example, if one makes predictions about the amount of rainfall each day for a whole month, and for each day, you make a predictive statement, "the amount of rainfall today will not be more than x inches, with 50% chance", and if indeed, rainfall did not exceed x inches for roughly 15 out of 30 days (50% of the days), then the predictions are said to be calibrated. We will explore calibration more and show how it can be visualized in the next section.
Sharpness refers to the concentration of the predictive distributions and is a property of these distributions only (i.e. doesn’t depend on ground truth observations) (Gneiting et al.). Sharpness is evaluated solely based on the predictive distribution, and neither the datapoint nor the ground truth distribution are considered when measuring sharpness. As an example, a Gaussian distributional prediction with mean 1 and variance 0.5 is sharper than a prediction with mean 1 and variance 3. Sharpness is desired because ideally, the predictive distribution should be tight around the observed data. 

One class of metrics that considers both calibration and sharpness simultaneously is proper scoring rules. According to the seminal work by [Gneiting and Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), a proper scoring rule is any function (with mild conditions) that assigns a score to a predictive probability distribution, where the maximum score of the function is attained when the predictive distribution exactly matches the ground truth distribution (i.e. the distribution of the data). Examples of proper scoring rules include log likelihood, the Brier score, continuous ranked probability score (CRPS), and the check score (a.k.a. pinball loss).

Uncertainty Toolbox provides functionalities to easily compute these metrics. In the next section, we give a concrete example of how to use our toolbox to evaluate the quantification of predictive uncertainty.


## Uncertainty Toolbox in Action
To demonstrate Uncertainty Toolbox, we step through a workflow in which we train a simple model, visualize its outputs, compute evaluation metrics, and recalibrate it for better performance. Previously, this would have had to be done through an amalgamation of different libraries and/or code written from scratch. Readers are encouraged to see this [Colab tutorial notebook](https://colab.research.google.com/drive/1fg61MFmVmgFYM4CRm_aoiZ_WRdxbIhIe?usp=sharing) for the full code.

For this example we will focus on a 1D regression problem with synthetic data. The data, which is visualized in the plot below, has heteroskedastic uniform noise.

<p align="center"><img src="/assets/tutorial/toydata.png" width=500 /></p>

To learn the conditional distribution Y given X, we will use a neural network model that outputs the mean and standard deviation of a Gaussian distribution (often called a Probabilistic Neural Net (PNN)). This model has been shown to have good performance, especially when ensembled ([Chua et al. 2018](https://arxiv.org/abs/1805.12114); [Lakshminarayanan et al. 2017](https://arxiv.org/abs/1612.01474)). However, the performance may suffer here since the true noise distribution is uniform instead of Gaussian. We will skip many of the details of the model and training since Uncertainty Toolbox is focused around model evaluation rather than model learning.

After training the model, we can use Uncertainty Toolbox’s visualizations to sanity check our predictions. Since this is a 1D regression problem, we can visualize the predicted distribution alongside the test data.

```python
# Plot confidence bands on test data
uct.viz.plot_xy(pred_mean, pred_std, te_y, te_x)
```

<p align="center"><img src="/assets/tutorial/confband_plot.png" width=400 /></p>

There are additional useful ways of visualizing model predictions outside of the 1D regression problem. We can also sort the values of the true observed y values in the test set (orange dashed line below) and plot the predicted mean values alongside them (solid blue dots). The 95% prediction interval is given along with the predicted mean values so we can see whether the true observations fall inside the prediction intervals.

```python
# Plot ordered prediction intervals
uct.viz.plot_intervals_ordered(pred_mean, pred_std, te_y)
```

<p align="center"><img src="/assets/tutorial/ordintv_plot.png" width=400 /></p>

One of the most important plots to evaluate predictive uncertainty of a model is the average calibration plot. When making a prediction, one can form an α-prediction interval that aims to capture observed values α% of the time. We can iterate over values of α and see the proportion of the test data that actually fall within the prediction interval. The calibration plot then shows the predicted proportion of the test data we expect to lie inside the interval on the x-axis and the observed proportion of the test data inside the interval on the y-axis.

A perfectly calibrated model will produce the line f(x) = x. We can use the area between the produced curve and the f(x) = x line to gauge how miscalibrated our model is. By looking at the calibration plot below for our model, we can see that it is slightly miscalibrated by being overconfident. That is, our model often produces predictive distributions that are too narrow.

```python
# Plot average calibration
uct.viz.plot_calibration(pred_mean, pred_std, te_y)
```

<p align="center"><img src="/assets/tutorial/undercal_plot.png" width=400 /></p>

The above plot gives information about our model’s average calibration, i.e. to produce the plot, we consider the prediction intervals across the entire test set. While this is an important indicator of our model’s performance, it does not guarantee that our model is correct. A truly correct model would be [individually calibrated](https://arxiv.org/abs/2006.10288), but individual calibration usually cannot be measured with a finite dataset.

Instead, we can use adversarial group calibration, a metric that was introduced by [Zhao et. al.](https://arxiv.org/abs/2006.10288). This involves taking many random subsets of the test data, computing miscalibration on each subset, and then reporting the worst miscalibration across the subsets. The plot below shows this metric as we vary the size of the subsets constructed (x-axis). For each subset size the procedure is repeated several times and the shaded region shows the standard error. Note that an individually calibrated model should have low calibration error for any group size.

```python
# Plot adversarial group calibration
uct.viz.plot_adversarial_group_calibration(pred_mean, pred_std, te_y)
```

<p align="center"><img src="/assets/tutorial/advcal_plot.png" width=500 /></p>

Alongside visualizations, Uncertainty Toolbox can be used to compute a suite of metrics given a set of test data, predicted means, and predicted standard deviations. These include accuracy and fit metrics (e.g. RMSE, MAE, and MDAE), calibration metrics, sharpness (average width of prediction intervals), and proper scoring rule metrics. The results are stored and returned in a dictionary object, and can also be printed out as shown below:

```python
# Get all metrics
pnn_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, te_y)
```

<p align="center"><img src="/assets/tutorial/metrics_printout.png" width=400 /></p>

Finally, Uncertainty Toolbox provides a way to perform recalibration. Assuming that there is additional validation data set aside, the outputted prediction intervals can be adjusted so that the model has better average calibration. The algorithm that our toolbox implements is one introduced by [Kuleshov et. al.](https://arxiv.org/abs/1807.00263) which relies on isotonic regression. After recalibrating our model and measuring calibration on the test set, we can see that our model produces wider prediction intervals and is therefore better calibrated.


<p align="center"><img src="/assets/tutorial/recal_plot.png" width=400 /></p>


## Conclusion

In this blog post we briefly summarized some key concepts in uncertainty and walked through an example use of Uncertainty Toolbox. We hope that this toolbox is useful for accelerating and uniting research efforts for uncertainty in machine learning. Looking towards the future, we plan to expand the scope to cover additional types of models and settings.


## References
```
[0] Gneiting, Tilmann, and Adrian E. Raftery. "Strictly proper scoring rules, prediction, and estimation." Journal of the American statistical Association 102.477 (2007): 359-378.

[1] Chua, Kurtland, et al. "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models." NeurIPS. 2018.

[2] Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." NeurIPS. 2017.

[3] Zhao, Shengjia, Tengyu Ma, and Stefano Ermon. "Individual calibration with randomized forecasting." International Conference on Machine Learning. PMLR, 2020.

[4] Kuleshov, Volodymyr, Nathan Fenner, and Stefano Ermon. "Accurate uncertainties for deep learning using calibrated regression." International Conference on Machine Learning. PMLR, 2018.
```

