import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc
import sqlite3
import collections
import six
import seaborn


def is_iterable(arg):
    return isinstance(arg, collections.Iterable) and not isinstance(arg, six.string_types)


def calculate_sensitivity(prediction, truth):
    try:
        true_positives = np.logical_and(truth, prediction)
        return np.count_nonzero(true_positives) / np.count_nonzero(truth)
    except ZeroDivisionError:
        return 0


def calculate_false_positive_rate(prediction, truth):
    false_positives = np.logical_and(truth == False, prediction)
    return np.count_nonzero(false_positives) / np.count_nonzero(truth == False)


def calculate_specificity(prediction, truth):
    true_negatives = np.logical_and(truth == False, prediction == False)
    return np.count_nonzero(true_negatives) / np.count_nonzero(truth == False)


def calculate_precision(prediction, truth):
    """
    i.e. Positive predictive value
    """
    true_positives = np.count_nonzero(np.logical_and(truth, prediction))
    false_positives = np.count_nonzero(np.logical_and(truth == False, prediction))
    if true_positives or false_positives:
        return true_positives / (true_positives + false_positives)


def calculate_positive_predictive_value(prediction, truth):
    return calculate_precision(prediction, truth)


def calculate_negative_predictive_value(prediction, truth):
    true_negatives = np.count_nonzero(np.logical_and(truth == False, prediction == False))
    false_negatives = np.count_nonzero(np.logical_and(truth, prediction == False))
    if true_negatives or false_negatives:
        return true_negatives / (true_negatives + false_negatives)


def calculate_accuracy(prediction, truth):
    return np.count_nonzero(truth == prediction) / len(truth)


def calculate_f1_score(prediction, truth):
    try:
        recall = calculate_sensitivity(prediction, truth)
        precision = calculate_precision(prediction, truth)
        return (2 * recall * precision) / (recall + precision)
    except ZeroDivisionError:
        return 0


def calculate_prediction_metrics(prediction, truth):
    return (calculate_accuracy(prediction, truth),
            calculate_precision(prediction, truth),
            calculate_negative_predictive_value(prediction, truth),
            calculate_sensitivity(prediction, truth),
            calculate_specificity(prediction, truth))


def calculate_roc_characteristics(values, thresholds, truth):
    tpr = list()
    fpr = list()
    for threshold in thresholds:
        positives = values <= threshold
        tpr.append(calculate_sensitivity(positives, truth))
        fpr.append(calculate_false_positive_rate(positives, truth))
    return tpr, fpr


def calculate_auc(values, thresholds, truth):
    tpr, fpr = calculate_roc_characteristics(values, thresholds, truth)
    return auc(fpr, tpr)


def extract_modality_data(dataframe, truth, column, ignore=None):
    if is_iterable(column):
        column_names = column
    else:
        column_names = [column]
    values, target = extract_distances(dataframe=dataframe, column_names=column_names, truth=truth, exclude=ignore)
    step = (values.max() - values.min())/1000
    thresholds = np.arange(values.min() - step, values.max() + step, step)
    return values, thresholds, target


def extract_distances(dataframe, column_names, truth, exclude=None):
    _columns = np.zeros(len(dataframe.columns), dtype=np.bool)
    column_names = np.asarray(column_names)

    for column_index, column in enumerate(dataframe.columns.values):
        if column in column_names:
            _columns[column_index] = True

    _distances = np.asarray(dataframe.values.T[_columns].T, dtype=np.float)
    _distances[np.isnan(_distances)] = np.inf
    min_distances = _distances.min(axis=1)
    min_modalities = column_names[_distances.argmin(axis=1)]

    # I want to exclude any points that do not have matching modalities
    _exclude = logical_or([min_distances == np.inf,
                           np.logical_and(truth,
                                          logical_and([np.isnan(dataframe[column].values)
                                                       for column in column_names]))])

    if exclude is not None:
        _exclude = np.logical_or(_exclude, exclude)

    _index = _exclude == False
    min_distances = min_distances[_index]
    truth = truth[_index]
    return min_distances, truth


def plot_roc_curve(data, lw=2, title=None, legend=True, diagonal=True, show_auc=False, show_plot=True):
    fig = plt.figure()
    for name, data_value in data.items():
        tpr, fpr, n, color = data_value
        if show_auc:
            label = '{0} (AUC={1:0.2f})'.format(name, auc(fpr, tpr))
        else:
            label = name
        plt.plot(fpr, tpr, color=color, lw=lw, label=label)
    if diagonal:
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    if title:
        plt.title(title)
    if legend:
        plt.legend(loc="lower right")
    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig


def plot_roc_curve_from_columns(dataframe, column_names, truth, title=None, colors=None, show_auc=False, legend=True,
                                labels=None, show_plot=True):
    roc_data = dict()
    if not colors:
        colors = [None] * len(column_names)
    for index, column_name in enumerate(column_names):
        color = colors[index]
        if labels:
            label = labels[index]
        else:
            label = column_name
        values, thresholds, truth_subset = extract_modality_data(dataframe, truth, column_name)
        if np.count_nonzero(truth_subset) > 0:
            tpr, fpr = calculate_roc_characteristics(values=values, thresholds=thresholds, truth=truth_subset)
            roc_data[label] = (tpr, fpr, len(values), color)
    return plot_roc_curve(roc_data, title=title, show_auc=show_auc, legend=legend, show_plot=show_plot)


def compute_prediction_metrics_by_column(dataframe, truth, column, thresholds):
    records = list()
    metric_names = ["Accuracy", "Precision (PPV)", "NPV", "Sensitivity", "Specificity"]
    header = ["Distance (mm)"] + metric_names
    values, _, target = extract_modality_data(dataframe, truth, column)
    for threshold in thresholds:
        records.append([threshold] + list(calculate_prediction_metrics(values <= threshold, target)))
    return pd.DataFrame.from_records(records, columns=header, index=header[0])


def plot_predictive_value_by_column(dataframe, truth, column, thresholds, title=None):
    values, _, target = extract_modality_data(dataframe, truth, column)
    ppv = list()
    npv = list()
    for threshold in thresholds:
        prediction = values <= threshold
        ppv.append(calculate_positive_predictive_value(prediction, target))
        npv.append(calculate_negative_predictive_value(prediction, target))
    return plot_predictive_values(ppv, npv, thresholds, title=title)


def plot_predictive_values(ppv, npv, thresholds, title=None, legend=True, lw=2,
                           ylabel='Predictive Value', xlabel='Geodesic Distance (mm)',
                           labels=("Positive", "Negative"), accuracy=None, f1=None):
    fig = plt.figure()
    plt.plot(thresholds, ppv, lw=lw, label=labels[0])
    plt.plot(thresholds, npv, lw=lw, label=labels[1])
    plt.ylim([0.0, 1.0])
    plt.xlim([thresholds.min(), thresholds.max()])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if accuracy:
        plt.plot(thresholds, accuracy, label="Accuracy", color="k", linestyle=":")
    if f1:
        plt.plot(thresholds, f1, label="F1 Score", color="k", linestyle="--")
    if title:
        plt.title(title)
    if legend:
        plt.legend(loc="lower right")
    plt.show()
    return fig


def plot_sensitivity_and_specificity_by_column(dataframe, truth, column, thresholds,
                                               title=None, xmin=0):
    values, _, target = extract_modality_data(dataframe, truth, column)
    sensitivity = list()
    specificity = list()
    accuracy = list()
    for threshold in thresholds:
        prediction = values <= threshold
        sensitivity.append(calculate_sensitivity(prediction, target))
        specificity.append(calculate_specificity(prediction, target))
        accuracy.append(calculate_accuracy(prediction, target))
    fig = plt.figure()
    plt.plot(thresholds, sensitivity, lw=2, label="Sensitivity")
    plt.plot(thresholds, specificity, lw=2, label="Specificity")
    plt.ylim([0.0, 1.0])
    plt.xlim([xmin, thresholds.max()])
    plt.xlabel("Geodesic Distance (mm)")
    plt.plot(thresholds, accuracy, label="Accuracy", linestyle="--", color="k")
    arg_max = np.argmax(accuracy)
    max_value = accuracy[arg_max]
    max_distance = thresholds[arg_max]
    plt.axvline(max_distance,
                label="Max Accuracy ({0:.2f}, {1:.1f}mm)".format(max_value, max_distance),
                color="k", linestyle=":")
    if title:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    return fig


def plot_sensitivity_vs_specificity(distances, thresholds, target, label=None, title=None):
    sensitivity, specificity = compute_sensitivity_and_specificity_arrays(distances, thresholds, target)
    fig = plt.figure()
    plt.plot(specificity, sensitivity, lw=2, label=label)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.ylabel("Sensitivity")
    plt.xlabel("Specificity")
    if title:
        plt.title(title)
    plt.show()
    return fig


def compute_sensitivity_and_specificity_arrays(distances, thresholds, target):
    sensitivity = list()
    specificity = list()
    for threshold in thresholds:
        prediction = distances <= threshold
        sensitivity.append(calculate_sensitivity(prediction, target))
        specificity.append(calculate_specificity(prediction, target))
    return np.asarray(sensitivity), np.asarray(specificity)


def plot_comparative_sensitivity_vs_specificity(distance_arrays, thresholds, target_arrays, labels=None,
                                                title=None):
    fig = plt.figure()
    for index, (distances, target) in enumerate(zip(distance_arrays, target_arrays)):
        sensitivity, specificity = compute_sensitivity_and_specificity_arrays(distances=distances,
                                                                              thresholds=thresholds,
                                                                              target=target)
        if labels:
            label = labels[index]
        else:
            label = None
        plt.plot(specificity, sensitivity, lw=2, label=label)
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.00])
    plt.ylabel("Sensitivity")
    plt.xlabel("Specificity")
    if title:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    return fig


def plot_positive_vs_negative_predictive_value(distances, thresholds, target, label=None, title=None):
    ppv = list()
    npv = list()
    for threshold in thresholds:
        prediction = distances <= threshold
        ppv.append(calculate_positive_predictive_value(prediction, target))
        npv.append(calculate_negative_predictive_value(prediction, target))
    fig = plt.figure()
    plt.plot(ppv, npv, lw=2, label=label)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.ylabel("Negative Predictive Value")
    plt.xlabel("Positive Predictive Value")
    if title:
        plt.title(title)
    plt.show()
    return fig


def compare_columns(dataframe, truth, columns, thresholds, title="Accuracy",
                    metric=calculate_accuracy, show_max=False, xmin=0):
    fig = plt.figure()
    if title:
        plt.title(title)
    for column in columns:
        values, _, target = extract_modality_data(dataframe, truth, column)
        measures = list()
        for threshold in thresholds:
            prediction = values <= threshold
            measures.append(metric(prediction, target))
        if "bold" in column:
            key = "fMRI"
        else:
            key = "MEG"
        key = " ".join((key, column.split("_")[1]))
        line, = plt.plot(thresholds, measures, label=key)
        if show_max:
            max_value = np.max(measures)
            max_index = np.argmax(measures)
            max_distance = thresholds[max_index]
            plt.axvline(max_distance,
                        label=key + " Max ({0:.2f}, {1:.1f}mm)".format(max_value,
                                                                       max_distance),
                        color=line.get_color(), linestyle=":")
    plt.xlabel("Geodesic Distance (mm)")
    plt.xlim([xmin, thresholds[-1]])
    plt.ylim([0, 1.05])
    plt.legend(loc="lower right")
    plt.show()
    return fig


def load_point_categories(dataframe, db_file):
    db_connection = load_db(db_file)
    categories = list()
    db_cursor = db_connection.cursor()
    for row_index, row in dataframe.iterrows():
        point_name = row['point_name']
        if 'negative' in point_name.lower():
            category = 'N'
        else:
            category = get_point_category(db_cursor, point_name, row['subject_id'],
                                          row['surgery_id'])
        categories.append(category)
    return np.asarray(categories)


def get_point_category(db_cursor, point_name, subject_id, surgery_id):
    cmd = ("SELECT Point.category FROM Point,PointSet,Surgery,Subject "
           "WHERE Point.point_set_id=PointSet.id "
           "AND Point.category is not NULL "
           "AND PointSet.surgery_id=Surgery.id "
           "AND Surgery.subject_id=Subject.id "
           "AND Surgery.name='{surgery_id:02d}' "
           "AND Subject.id={subject_id} "
           "AND Point.name='{point_name}'".format(subject_id=subject_id,
                                                  surgery_id=surgery_id,
                                                  point_name=point_name))
    db_cursor.execute(cmd)
    results = db_cursor.fetchall()
    assert len(results) == 1
    return results[0][0]


def load_db(db_file):
    return sqlite3.connect(db_file)


def logical_and(array_list):
    array = array_list[0]
    for other_array in array_list[1:]:
        array = np.logical_and(array, other_array)
    return array


def logical_or(array_list):
    array = array_list[0]
    for other_array in array_list[1:]:
        array = np.logical_or(array, other_array)
    return array


def get_index_value(iterable, index):
    if iterable:
        return iterable[index]


def plot_sensitivity_and_specificity_by_columns(dataframe, targets, columns, thresholds, labels=None,
                                                title=None, exclusions=None, lw=2, sensitivity_pattern="-",
                                                specificity_pattern="--", ylim=(0, 1.05), convert_to_cm=False,
                                                xlabel="Euclidean Distance ({units})", legend=True, units="mm",
                                                legend_loc="lower right", exclude=None, confidence_interval=False,
                                                confidence_interval_alpha=0.15, subplots=None, same_spec_color=True,
                                                show_j_stat=False, j_stat_color="black", legend_columns=1,
                                                figsize=None, legend_bbox=None, legend_fontsize='medium',
                                                x_label_offset=0.01, y_label_offset=0.01, label_max_j_stat=True,
                                                max_j_stat_fontsize="small", sensitivity_color='C0',
                                                specificity_color='C2', print_distance=None, sharex=True,
                                                sharey=True):
    lines = dict()
    if exclude is not None and exclusions is None:
        exclusions = [exclude] * len(columns)
    if subplots:
        fig, axes = plt.subplots(*subplots, sharex=sharex, sharey=sharey, figsize=figsize)
        axes = np.ravel(axes)
    else:
        fig, ax = plt.subplots(figsize=figsize)
    if '{units}' in xlabel:
        xlabel = xlabel.format(units=units)
    if convert_to_cm:
        x = thresholds / 10
    else:
        x = thresholds
    for index, column_name in enumerate(columns):
        ignore_index = get_index_value(exclusions, index)
        label = get_index_value(labels, index)
        truth = targets[index]
        values, _, truth_subset = extract_modality_data(dataframe,
                                                        truth,
                                                        column_name,
                                                        ignore_index)
        if subplots:
            ax = axes[index]
            ax.set_title(label)
            ax.set_xlabel(xlabel)
            ax.set_xlim([x.min(), x.max()])
            ax.set_ylim(ylim)

        sensitivity, specificity = compute_sensitivity_and_specificity_arrays(values, thresholds, truth_subset)
        if same_spec_color:
            sens_line = ax.plot(x, sensitivity, sensitivity_pattern, lw=lw, label=label)[0]
            spec_line = ax.plot(x, specificity, specificity_pattern, lw=lw, color=sens_line.get_color())[0]
        else:
            sens_line = ax.plot(x, sensitivity, sensitivity_pattern, lw=lw, label="Sensitivity",
                                color=sensitivity_color)[0]
            spec_line = ax.plot(x, specificity, specificity_pattern, lw=lw, label="Specificity",
                                color=specificity_color)[0]
        if sens_line.get_label() not in lines:
            lines[sens_line.get_label()] = sens_line
        if spec_line.get_label() and spec_line.get_label not in lines:
            lines[spec_line.get_label()] = spec_line
        if confidence_interval:
            n_positives = np.count_nonzero(truth_subset)
            w_minus, w_plus = wilson_score_interval_with_continuity_correction(sensitivity, n_positives)
            ax.fill_between(x, w_minus, w_plus, alpha=confidence_interval_alpha, facecolor=sens_line.get_color())
            n_negatives = len(truth_subset) - n_positives
            w_minus, w_plus = wilson_score_interval_with_continuity_correction(specificity, n_negatives)
            ax.fill_between(x, w_minus, w_plus, alpha=confidence_interval_alpha, facecolor=spec_line.get_color())
        if show_j_stat:
            j_stat = sensitivity + specificity - 1
            j_line = ax.plot(x, j_stat, color=j_stat_color, label="J statistic")[0]
            if j_line.get_label() not in lines:
                lines[j_line.get_label()] = j_line
            if label_max_j_stat:
                i = j_stat.argmax()
                dist = x[i]
                print("{}:\tDistance: {:.1f}{units}\tSensitivity: {:.2f}\tSpecificity: {:.2f}".format(label,
                                                                                                      dist,
                                                                                                      sensitivity[i],
                                                                                                      specificity[i],
                                                                                                      units=units))
                ax.annotate('{:.1f}mm'.format(dist), 
                            xy=(dist + x_label_offset,
                                y_label_offset),
                            color=j_stat_color, 
                            fontsize=max_j_stat_fontsize,
                            rotation=0)
                max_j_stat_line = ax.axvline(dist, color=j_stat_color, linestyle='--', label='Maximum J Statistic')
                if max_j_stat_line.get_label() not in lines:
                    lines[max_j_stat_line.get_label()] = max_j_stat_line
        if print_distance:
            dist = print_distance
            i = np.squeeze(np.where(x == dist))
            print("{}:\tDistance: {:.1f}{units}\tSensitivity: {:.2f}\tSpecificity: {:.2f}".format(label,
                                                                                                  dist,
                                                                                                  sensitivity[i],
                                                                                                  specificity[i],
                                                                                                  units=units))

    if same_spec_color:
        _sens = ax.plot(thresholds, thresholds, sensitivity_pattern, color='k', label="Sensitivity")[0]
        _spec = ax.plot(thresholds, thresholds, specificity_pattern, color='k', label="Specificity")[0]

    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim(ylim)
    if legend:
        if subplots:
            legend_lines = list(lines.values())
            legend_labels = list(lines.keys())
            ax = axes[0]
            if legend_bbox is None:
                ax.legend(legend_lines, legend_labels, loc=legend_loc, ncol=legend_columns, fontsize=legend_fontsize,
                          bbox_transform=fig.transFigure)
            else:
                ax.legend(legend_lines, legend_labels, loc=legend_loc, ncol=legend_columns, bbox_to_anchor=legend_bbox,
                          fontsize=legend_fontsize, bbox_transform=fig.transFigure)
        else:
            ax.legend(loc=legend_loc, ncol=legend_columns)
    if title:
        ax.set_title(title)
    if same_spec_color:
        _sens.set_visible(False)
        _spec.set_visible(False)
    ax.set_xlabel(xlabel)
    return fig


def plot_positive_and_negative_predictive_values_by_columns(dataframe, columns, thresholds, targets, exclusions=None,
                                                            title=None, xlabel="Euclidean Distance (cm)",
                                                            legend=True, convert_to_cm=True, labels=None,
                                                            ylim=(0, 1.05), n_legend_columns=1,
                                                            legend_loc="lower right"):
    fig, ax = plt.subplots()
    if convert_to_cm:
        x = thresholds / 10
    else:
        x = thresholds
    for index, column in enumerate(columns):
        label = get_index_value(labels, index)
        exclude = get_index_value(exclusions, index)
        truth = targets[index]
        _npv, _ppv = get_ppv_and_npv_values(dataframe, column, thresholds, truth, exclude=exclude)
        _ppv_plot = ax.plot(x, _ppv, label=label)[0]
        _npv_plot = ax.plot(x, _npv, '--', color=_ppv_plot.get_color())

    positive = ax.plot(thresholds, thresholds * -1, color='k', label='PPV')[0]
    negative = ax.plot(thresholds, thresholds * -1, '--', color='k', label='NPV')[0]

    if legend:
        ax.legend(loc=legend_loc, ncol=n_legend_columns)
    if title:
        ax.set_title(title)

    ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Predictive Value')
    ax.set_xlim([x.min(), x.max()])
    return fig


def get_ppv_and_npv_values(dataframe, column_names, thresholds, truth, exclude=None):
    _columns = np.zeros(len(dataframe.columns), dtype=np.bool)
    column_names = np.asarray(column_names)

    for column_index, column in enumerate(dataframe.columns.values):
        if column in column_names:
            _columns[column_index] = True

    _distances = np.asarray(dataframe.values.T[_columns].T, dtype=np.float)
    _distances[np.isnan(_distances)] = np.inf
    min_distances = _distances.min(axis=1)
    min_modalities = column_names[_distances.argmin(axis=1)]

    # I want to exclude any points that do not have matching modalities
    _exclude = logical_or([min_distances == np.inf,
                           np.logical_and(truth,
                                          logical_and([np.isnan(dataframe[column].values)
                                                       for column in column_names]))])

    if exclude is not None:
        _exclude = np.logical_or(_exclude, exclude)

    _index = _exclude == False

    min_distances = min_distances[_index]
    truth = truth[_index]

    _ppv = list()
    _npv = list()

    for cutoff in thresholds:
        _prediction = (min_distances) <= cutoff
        _ppv.append(calculate_positive_predictive_value(prediction=_prediction, truth=truth))
        _npv.append(calculate_negative_predictive_value(prediction=_prediction, truth=truth))

    return np.asarray(_npv), np.asarray(_ppv)


def wilson_score_interval_with_continuity_correction(score, sample_size, z=1.96):
    a = 2 * sample_size * score + z**2  # 2np + z^2
    b = z**2 - (1/float(sample_size)) + 4 * sample_size * score * (1 - score)  # z^2 - 1/n + 4np(1-p)
    c = 4 * score - 2  # 4p - 2
    d = 2 * (sample_size + z**2)  # 2(n + z^2)

    w_minus = (a - (z * np.sqrt(b + c) + 1))/d
    w_plus = (a + (z * np.sqrt(b - c) + 1))/d

    # perform continuity correction
    if isinstance(w_minus, np.ndarray):
        w_minus = np.max([w_minus, np.zeros(len(w_minus))], axis=0)
        w_plus = np.min([w_plus, np.ones(len(w_plus))], axis=0)
    else:
        w_minus = np.max([0, w_minus])
        w_plus = np.min([1, w_plus])
    return w_minus, w_plus


def get_results(labels, predictions, truths):
    results = {"label": [], "measurement": [], "value": []}
    confidence_intervals = {"labels": labels,
                            "PPV": [], "NPV": [],
                            "Specificity": [], "Sensitivity": []}

    for label, prediction, truth in zip(labels, predictions, truths):
        ppv = calculate_positive_predictive_value(prediction=prediction, truth=truth)
        n_predicted_positve = np.count_nonzero(prediction)
        w_minus, w_plus = wilson_score_interval_with_continuity_correction(ppv, n_predicted_positve)
        confidence_intervals["PPV"].append((w_minus, w_plus))

        npv = calculate_negative_predictive_value(prediction=prediction, truth=truth)
        n_predicted_negative = np.count_nonzero(prediction == False)
        w_minus, w_plus = wilson_score_interval_with_continuity_correction(npv, n_predicted_negative)
        confidence_intervals["NPV"].append((w_minus, w_plus))

        n_positives = np.count_nonzero(truth)
        n_negatives = np.count_nonzero(truth == False)

        sensitivity = calculate_sensitivity(prediction=prediction, truth=truth)
        w_minus, w_plus = wilson_score_interval_with_continuity_correction(sensitivity, n_positives)
        confidence_intervals["Sensitivity"].append((w_minus, w_plus))

        specificity = calculate_specificity(prediction=prediction, truth=truth)
        w_minus, w_plus = wilson_score_interval_with_continuity_correction(specificity, n_negatives)
        confidence_intervals["Specificity"].append((w_minus, w_plus))

        results["value"].append(calculate_positive_predictive_value(prediction=prediction, truth=truth))
        results["label"].append(label)
        results["measurement"].append("PPV")
        results["value"].append(calculate_negative_predictive_value(prediction=prediction, truth=truth))
        results["label"].append(label)
        results["measurement"].append("NPV")
        results["value"].append(calculate_sensitivity(prediction=prediction, truth=truth))
        results["label"].append(label)
        results["measurement"].append("Sensitivity")
        results["value"].append(calculate_specificity(prediction=prediction, truth=truth))
        results["label"].append(label)
        results["measurement"].append("Specificity")

    results_df = pd.DataFrame(results)
    return results_df, confidence_intervals


def plot_results(results_df, confidence_intervals, order=("NPV", "PPV", "Specificity", "Sensitivity"), xlabel="",
                 ylabel="", ylim=(0, 1.0), ci_color="k", x_column="measurement", hue_column="label", y_column="value",
                 ci_skip=("AUC",), legend_loc="upper center", legend_ncol=3, legend_bbox=(0.5, 1.2)):
    fig, ax = plt.subplots()
    seaborn.barplot(data=results_df, x=x_column, hue=hue_column, y=y_column, ax=ax, order=order)
    index = 0
    lines = ax.get_lines()

    for i in range(3):
        for label in order:
            if label not in ci_skip:
                x = lines[index].get_xdata()
                y = confidence_intervals[label][i]
                ax.errorbar(x=x, y=y, color=ci_color)
            index += 1
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.legend(loc=legend_loc, ncol=legend_ncol, bbox_to_anchor=legend_bbox)
    return fig
