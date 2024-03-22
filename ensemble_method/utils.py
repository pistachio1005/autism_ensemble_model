from sklearn.metrics import *#accuracy_score#, confusion_matrix, recall_score, precision_score
from fairlearn.metrics import count, false_negative_rate, false_positive_rate, selection_rate
from fairlearn.metrics import MetricFrame 
from sklearn.utils import check_consistent_length
import numpy as np
from fairlearn.experimental.enable_metric_frame_plotting import plot_metric_frame

#Code wrriten for bmi 223 with functions from fairlearn example files

def general_wilson(p, n, digits=4, z=1.959964):
    """Return lower and upper bound of a Wilson confidence interval.

    Parameters
    ----------
    p : float
        Proportion of successes.
    n : int
        Total number of trials.
    digits : int
        Digits of precision to which the returned bound will be rounded
    z : float
        Z-score, which indicates the number of standard deviations of confidence.
        The default value of 1.959964 is for a 95% confidence interval

    Returns
    -------
        np.ndarray
        Array of length 2 of form: [lower_bound, upper_bound]
    """
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z * z / (2 * n)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z * z / (4 * n))) / np.sqrt(n)
    lower_bound = (
        centre_adjusted_probability - z * adjusted_standard_deviation
    ) / denominator
    upper_bound = (
        centre_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator
    return np.array([round(lower_bound, digits), round(upper_bound, digits)])


def recall_wilson(y_true, y_pred, digits=4, z=1.959964):
    """Return a Wilson confidence interval for the recall metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels
    y_pred : array-like of shape (n_samples,)
        Predicted labels

    Returns
    -------
        np.ndarray
        Array of length 2 of form: [lower_bound, upper_bound]
    """
    check_consistent_length(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    bounds = general_wilson(tp / (tp + fn), tp + fn, digits, z)
    return bounds


def accuracy_wilson(y_true, y_pred, digits=4, z=1.959964):
    """Return a Wilson confidence interval for the accuracy metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels
    y_pred : array-like of shape (n_samples,)
        Predicted labels

    Returns
    -------
        np.ndarray
        Array of length 2 of form: [lower_bound, upper_bound]
    """
    check_consistent_length(y_true, y_pred)
    score = accuracy_score(y_true, y_pred)
    bounds = general_wilson(score, len(y_true), digits, z)
    return bounds

def confidence_interval_plot(X_test, y_test, y_pred, label, interval=False):
    sensitive_test = X_test[label]
    metrics_dict = {
    "Recall": recall_score,
    "Recall Bounds": recall_wilson,
    "Accuracy": accuracy_score,
    "Accuracy Bounds": accuracy_wilson,
    }
    metric_frame = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_test
    )
    if not interval:
        plot_metric_frame(metric_frame, kind="point", metrics=["Recall", "Accuracy"])
    else:
        plot_metric_frame(
        metric_frame,
        kind="bar",
        metrics=["Recall", "Accuracy"],
        conf_intervals=["Recall Bounds", "Accuracy Bounds"],
        plot_ci_labels=True,
        subplots=False,
        )
        plot_metric_frame(  
        metric_frame,
        kind="point",
        metrics="Recall",
        conf_intervals="Recall Bounds",
        )   
    return


def metrics_plots(X_test, y_true, y_pred, label, save_path):
    sensitive_feat = X_test[label]
    metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "false positive rate": false_positive_rate,
    "false negative rate": false_negative_rate,
    "selection rate": selection_rate,
    "count": count, 
    }
    metric_frame = MetricFrame(
    metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_feat
    )
    metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
    )

# Customize plots with ylim
    metric_frame.by_group.plot(
    kind="bar",
    ylim=[0, 1],
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics with assigned y-axis range",    
    )

# Customize plots with colormap
    metric_frame.by_group.plot(
    kind="bar",
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    colormap="Accent",
    title="Show all metrics in Accent colormap",
    )
    fig.figure.savefig(save_path)
# Customize plots with kind (note that we are only plotting the "count" metric here because we are showing a pie chart)
    metric_frame.by_group[["count"]].plot(
    kind="pie",
    subplots=True,
    layout=[1, 1],
    legend=False,
    figsize=[12, 8],
    title="Show count metric in pie chart",
    )

# Saving plots
    fig = metric_frame.by_group[["count"]].plot(
    kind="pie",
    subplots=True,
    layout=[1, 1],
    legend=False,
    figsize=[12, 8],
    title="Show count metric in pie chart",
    )
    