---
layout: page
title: About
---

# About

**Uncertainty Toolbox**
> A python toolbox for predictive uncertainty quantification, calibration,
> [metrics, and visualization](#metrics). 
> Also: a [glossary of useful terms](https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/master/docs/glossary.md){:target="_blank"} and a collection
> of [relevant papers and references](https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/master/docs/paper_list.md){:target="_blank"}.

Many machine learning methods return predictions along with uncertainties of some form,
such as distributions or confidence intervals. This begs the questions: How do we
determine which predictive uncertanties are best? What does it mean to produce a _best_
or _ideal_ uncertainty? Are our uncertainties accurate and _well calibrated_?

Uncertainty Toolbox provides standard metrics to quantify and compare predictive
uncertainty estimates, gives intuition for these metrics, produces visualizations of
these metrics/uncertainties, and implements simple "re-calibration" procedures to
improve these uncertainties.  This toolbox currently focuses on regression tasks.


## Toolbox Contents

Uncertainty Toolbox contains:
* [Glossary](https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/master/docs/glossary.md){:target="_blank"} of terms related to predictive uncertainty
  quantification.
* [Metrics](#metrics) for assessing quality of predictive uncertainty estimates.
* [Visualizations](#visualizations) for predictive uncertainty estimates and metrics.
* [Recalibration](#recalibration) methods for improving the calibration of a predictor.
* Relevant [publications and references](https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/master/docs/paper_list.md){:target="_blank"} on metrics and methods.
* A [tutorial](/tutorial/) walking through how to use Uncertainty Toolbox.


## Installation

Uncertainty Toolbox requires Python 3.6+. To install, clone and `cd` into this repo, and run:
```
$ pip install -e .
```


## Quick Start

```python
import uncertainty_toolbox as uct

# Load an example dataset of 100 predictions, uncertainties, and observations
predictions, predictions_std, y, x = uct.data.synthetic_sine_heteroscedastic(100)

# Compute all uncertainty metrics
metrics = uct.metrics.get_all_metrics(predictions, predictions_std, y)
```
This example computes [metrics](#metrics) for a vector of predicted values
(`predictions`) and associated uncertainties (`predictions_std`, a vector of standard
deviations), taken with respect to a corresponding set of observed values `y`.


## Metrics

Uncertainty Toolbox provides a number of [metrics](uncertainty_toolbox/metrics.py) to
quantify and compare predictive uncertainty estimates. For example, the
[`get_all_metrics`](uncertainty_toolbox/metrics.py#L165) function will return:
1. __average calibration__: _mean absolute calibration error, root mean squared calibration error, miscalibration area_
2. __adversarial group calibration__: _mean absolute adversarial group calibration error, root mean squared adversarial group calibration error_
3. __sharpness__: _expected standard deviation_
4. __proper scoring rules__: _negative log-likelihood, continuous ranked probability score, check score, interval score_
5. __accuracy__: _mean absolute error, root mean squared error, median absolute error, coefficient of determination, correlation_


## Visualizations

The following plots are a few of the [visualizations](uncertainty_toolbox/viz.py)
provided by Uncertainty Toolbox. See [this example](examples/viz_synth_sine.py) for code
to reproduce these plots.

**Overconfident** (_too little uncertainty_)
<p align="center"><img src="/assets/uncertainty_toolbox/overconf.png" width=700 /></p>

**Underconfident** (_too much uncertainty_)
<p align="center"><img src="/assets/uncertainty_toolbox/underconf.png" width=700 /></p>

**Well calibrated**
<p align="center"><img src="/assets/uncertainty_toolbox/calibrated.png" width=700 /></p>

And here are a few of the calibration metrics for the above three cases:

|              | Mean absolute calibration error (MACE) | Root mean squared calibration error (RMSCE) | Miscalibration area (MA) |
| :----- | :-----: | :-----: | :-----: |
| Overconfident | 0.19429 | 0.21753 | 0.19625 |
| Underconfident | 0.20692 | 0.23003 | 0.20901 |
| Well calibrated | 0.00862 | 0.01040 | 0.00865 |


## Recalibration

The following plots show the results of a
[recalibration](uncertainty_toolbox/recalibration.py) procedure provided by Uncertainty
Toolbox, which transforms a set of predictive uncertainties to improve average
calibration. The algorithm is based on isotonic regression, as proposed by [Kuleshov et
al](docs/paper_list.md#calibration-sharpness-and-recalibration-in-deep-learning).

See [this example](examples/viz_recalibrate.py) for code to reproduce these plots.

**Recalibrating overconfident predictions**
<p align="center"><img src="/assets/uncertainty_toolbox/recal_over.png" width=700 /></p>

|              | Mean absolute calibration error (MACE) | Root mean squared calibration error (RMSCE) | Miscalibration area (MA) |
| :----- | :-----: | :-----: | :-----: |
| Before Recalibration | 0.19429 | 0.21753 | 0.19625 |
| After Recalibration | 0.01124 | 0.02591 | 0.01117 |

**Recalibrating underconfident predictions**
<p align="center"><img src="/assets/uncertainty_toolbox/recal_under.png" width=700 /></p>

|              | Mean absolute calibration error (MACE) | Root mean squared calibration error (RMSCE) | Miscalibration area (MA) |
| :----- | :-----: | :-----: | :-----: |
| Before Recalibration | 0.20692 | 0.23003 | 0.20901 |
| After Recalibration | 0.00157 | 0.00205 | 0.00132 |


## Citation

If you use this toolbox, please consider citing one of the papers that led to its
development:
```
@article{chung2020beyond,
  title={Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification},
  author={Chung, Youngseog and Neiswanger, Willie and Char, Ian and Schneider, Jeff},
  journal={arXiv preprint arXiv:2011.09588},
  year={2020}
}

@article{tran2020methods,
  title={Methods for comparing uncertainty quantifications for material property predictions},
  author={Tran, Kevin and Neiswanger, Willie and Yoon, Junwoong and Zhang, Qingyang and Xing, Eric and Ulissi, Zachary W},
  journal={Machine Learning: Science and Technology},
  volume={1},
  number={2},
  pages={025006},
  year={2020},
  publisher={IOP Publishing}
}
```


## Acknowledgments

Development of Uncertainty Toolbox is [supported by](docs/acknowledgments.md) the following organizations.
<p align="center"><img src="/assets/uncertainty_toolbox/acks.png" width=300 /></p>