"""Copyright (C) 2023 aurouet lucas

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA."""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
import random
from scipy import spatial
import matplotlib.pyplot as plt


def check_type(features, label):
    if type(features) != np.array and type(features) != np.ndarray:
        warnings.warn(
            f'feaures were of type {type(features)} and have been converted to numpy array. Conversion may have corrupted the data.')
        features = features.to_numpy()
    else:
        pass

    if type(label) != np.array and type(label) != np.ndarray:
        warnings.warn(
            f'labels were of type {type(label)} and have been converted to numpy array. Conversion may have corrupted the data.')
        label = label.to_numpy()
    else:
        pass

    if features.shape[0] != label.shape[0]:
        raise ValueError(
            f'Found inconsistent numbers of samples: features have {features.shape[0]}, label have {label.shape[0]}')

    n_rows = features.shape[0]
    n_cols = features.shape[1]

    for j in range(0, n_cols):
        for i in range(0, n_rows):
            if np.issubsctype(features[i, j], np.number):
                pass
            else:
                raise ValueError(
                    f'features must be of type int or float not of type {type(features[i, j])}')

    return features, label


def support_diff(dual_coef, sv_1, sv_2):
    # List that holds missing supports
    sv_diff = []
    # List that holds the index of missing supports
    sv_diff_idx = []

    # Loop over both lists of supports
    # And create a list of missing supports
    for support in sv_1:
        if support.tolist() not in sv_2.tolist():
            sv_diff.append(support)

    # Reshape the array of missing supports
    # For easier access
    sv_diff = np.array(sv_diff).reshape(-1, sv_1.shape[1])

    # Get the index of the missing support
    for support in sv_diff:
        sv_diff_idx.append(np.where(np.all(sv_1 == support, axis=1))[0][0])

    # Remove missing/additional supports from dual coef
    dual_coef = np.delete(dual_coef, sv_diff_idx, axis=1)

    # Return dual coeff only for supports that exist in both
    return np.ravel(dual_coef)


def distance_to_replacement(outlier, attrib, sv_reg, features, labels, outliers_idx):
    dist = []
    dist_to_class = []

    outlier_class = outlier[1]
    outlier_value = outlier[0]

    full_covar = attrib['data']['inv_covar']

    cov_mat_X = attrib[outlier_class]['inv_covar']
    center_X = attrib[outlier_class]['centroid']

    for label in attrib.keys():
        if label != 'data':
            center = attrib[label]['centroid']
            dist_to_class.append(m_dist(outlier_value, center, full_covar))

    dist_to_class_srt = sorted(dist_to_class)

    dist_to_own = m_dist(outlier_value, center_X, full_covar)

    scaling_factor = max(1, dist_to_own / dist_to_class_srt[0])

    for s_reg in sv_reg.tolist():
        sv_idx = np.where(np.all(features == s_reg, axis=1))[0][0]
        sv_reg_class = labels[sv_idx]

        if sv_reg_class == outlier_class and sv_idx not in outliers_idx:
            dist.append(m_dist(outlier_value, s_reg, cov_mat_X))

    return sorted(dist)[0] * scaling_factor, scaling_factor


def m_dist(to_vec, from_vec, cov_mat):

    dist = (to_vec - from_vec).T.dot(cov_mat).dot(to_vec - from_vec)

    return dist


def alpha_diff(vec_1, vec_2, Z_base, Z_reg):

    sym_diff = set(vec_1).symmetric_difference(set(vec_2))

    for item in sym_diff:
        if item in vec_1:
            vec_1 = np.delete(vec_1, np.where(vec_1 == item))

    pos = []
    alpha_dist = []

    for idx in vec_1:
        pos.append(np.where(vec_1 == idx)[0][0])

    Z_base_sym = Z_base[:, pos]
    Z_reg_sym = Z_reg[:, pos]

    if Z_base_sym.shape[1] != 0:
        for i in range(0, Z_base_sym.shape[1]):
            cos_dist = spatial.distance.cosine(Z_base_sym[:, i], Z_reg_sym[:, i])
            alpha_dist.append(cos_dist)
    else:
        alpha_dist.append(0)

    return sum(alpha_dist) / len(alpha_dist)


def centroid_estimate(X, centroid, kernel_params):
    if centroid == 'mean':
        return np.mean(X, axis=0)
    elif centroid == 'mode':
        kernel = KernelDensity(**kernel_params)
        kernel.fit(X)
        kernel_scores = np.exp(kernel.score_samples(X))
        mode_idx = kernel_scores.argmax()
        mode = X[mode_idx, :]
        return mode
    elif centroid == 'median':
        return np.median(X, axis=0)
    else:
        raise ValueError(f'{centroid} is not a correct argument. Please use \'mean\' or \'mode\'')


def random_quantile(array, q, steps):

    cov_mat = np.cov(array, rowvar=False)
    inv_cov_mat = np.linalg.matrix_power(cov_mat, -1)

    quantile = []
    dist = []

    for x_i in array:

        dist_x_i = []

        for step in range(0, steps):

            random_idx = random.sample(range(0, array.shape[0]), k=1)[0]
            x_j = array[random_idx]
            dist_x_i.append((x_i - x_j).T.dot(inv_cov_mat).dot(x_i - x_j))

        dist.append(np.mean(dist_x_i))

    quantile = np.quantile(dist, q)

    return quantile


def outlier_reg(features, label, classifier, centroid='median', kernel_params={}, k=10, q=0.99, steps=5):
    # =============================================================================
    # Initialization
    # =============================================================================

    # This dictionnary holds the centroids and covariances matrices estimates
    attrib = {}
    sub_dic = {}

    # Create a dictonnary to hold the results
    results = {'index': [],
               'scaling': [],
               'influence': [],
               'quantiles': [],
               'intercept_dist': [],
               'alpha_dist': []}

    # lists of largest distances from centroid
    top_k_idx = []
    top_k_dist = []

    # Create the scaler object
    scaler = MinMaxScaler()
    # Features scaling
    features = scaler.fit_transform(features)

    # Covariance matrix and centroid estimation
    # across the entire sample
    sub_dic['inv_covar'] = np.linalg.matrix_power(np.cov(features, rowvar=False), -1)
    sub_dic['centroid'] = centroid_estimate(features, centroid, kernel_params)
    attrib['data'] = sub_dic

    # =============================================================================
    # Mahalanobis distance from centroid
    # =============================================================================

    # For each class in our data
    for classes in np.unique(label):
        # reset variables
        sub_dic = {}
        dist_to_centroid = []

        # keep samples that belong to "classes"
        idx = np.where(label == classes)[0].tolist()
        class_features = features[idx, :]

        # covariance matrix estimation for classes
        sub_dic['inv_covar'] = np.linalg.matrix_power(np.cov(class_features, rowvar=False), -1)
        sub_dic['centroid'] = centroid_estimate(class_features, centroid, kernel_params)
        attrib[classes] = sub_dic

        # Mahalanobis Distance to centroid
        # for each sample i
        for i in range(0, class_features.shape[0]):
            X = class_features[i]
            dist = m_dist(X, attrib[classes]['centroid'], attrib[classes]['inv_covar'])
            dist_to_centroid.append(round(dist, 3))

        # Sort distances in ascending order
        dist_to_centroid, idx = [list(x) for x in zip(*sorted(zip(dist_to_centroid, idx)))]

        # Get the k largest distances and corresponding index
        top_k_idx.extend(idx[-k:])
        top_k_dist.extend(dist_to_centroid[-k:])

        results['index'] = top_k_idx

        # =============================================================================
        # distance quantile
        # =============================================================================

        # remove outliers from data
        clean_features = np.delete(features, top_k_idx, axis=0)
        clean_label = np.delete(label, top_k_idx)
        idx = np.where(clean_label == classes)[0].tolist()
        clean_features = clean_features[idx, :]

        # compute the qth quantile
        attrib[classes]['quantile'] = random_quantile(class_features, q, steps)

    # =============================================================================
    # Influence Measure
    # =============================================================================

    # loop over all the outliers
    for idx_to_remove, dist in zip(top_k_idx, top_k_dist):
        # save the outlier values and class
        outlier = [features[idx_to_remove], label[idx_to_remove]]

        # fit the original model
        model_base = classifier.fit(features, label)

        # dual coef & support vectors for base model
        Z_base = model_base.dual_coef_
        sv_base = model_base.support_vectors_
        sv_base_idx = model_base.support_
        b_base = model_base.intercept_

        # only if the outlier is a support vector
        if idx_to_remove in sv_base_idx:

            # remove the outlier
            features_copy = np.delete(features, idx_to_remove, axis=0)
            label_copy = np.delete(label, idx_to_remove)

            # Refit the model without the outlier
            model_reg = classifier.fit(features_copy, label_copy)

            # Extract decision function (dual coefs) and support vectors
            Z_reg = model_reg.dual_coef_
            sv_reg = model_reg.support_vectors_
            sv_reg_idx = model_reg.support_
            b_reg = model_reg.intercept_

            # Comparison of alpha coefficients
            # for supports in both models
            alpha = alpha_diff(sv_base_idx, sv_reg_idx, Z_base, Z_reg)

            # Intercept comparison between both models
            b_dist = spatial.distance.cosine(b_base, b_reg)

            # Distance to closest new support vector
            sum_d, scaling_factor = distance_to_replacement(
                outlier, attrib, sv_reg, features, label, top_k_idx)
        else:
            sum_d = 0
            scaling_factor = 1
            alpha = 0
            b_dist = 0

        # compare with quantile
        results['quantiles'].append(attrib[outlier[1]]['quantile'])

        # Save the results
        results['influence'].append(round(sum_d, 3))
        results['scaling'].append(round(scaling_factor, 2))
        results['alpha_dist'].append(round(alpha, 3))
        results['intercept_dist'].append(round(b_dist, 3))

    # Save in dataframe and return
    results_df = pd.DataFrame(results)

    return results_df


def multistage_npor(features, label, classifier, centroid='median', kernel_params={}, k=10, q=0.99, steps=10):

    # =============================================================================
    # init
    # =============================================================================

    # randomly initiate the outlier counter to a positive integer
    count_outliers = label.shape[0]
    loop = 0

    # results
    inf_points = {'index': [],
                  'influence': [],
                  f'{q}_quantile': [],
                  'res_alpha': [],
                  'res_b': []}

    # initiate an index that does not change when removing sample to match indices after removal
    fixed_index = list(range(0, len(label)))

    # Check data type and convert to ndarrays if necessary
    features, label = check_type(features, label)

    # while at least one outlier is found keep searching
    while count_outliers > 0 and loop < label.shape[0] / 2:

        # initiate dictionnary
        current_values = {'current_index': [],
                          'current_influence': [],
                          'current_quantile': [],
                          'current_alpha': [],
                          'current_b': []}

        # update counters
        count_outliers = 0
        loop += 1

        # outlier search
        outliers = outlier_reg(features, label, classifier, centroid, kernel_params, k, q, steps)

        # print progress
        print(f"-- step n° {loop} --")
        print(outliers)

        # for each outlier found
        # if influence > quantile
        # save the values in inf_points
        for i in range(0, len(outliers)):
            if outliers['influence'].iloc[i] > outliers['quantiles'].iloc[i]:
                count_outliers += 1

                current_values['current_index'].append(outliers['index'].iloc[i])
                current_values['current_influence'].append(outliers['influence'].iloc[i])
                current_values['current_quantile'].append(outliers['quantiles'].iloc[i])
                current_values['current_alpha'].append(outliers['alpha_dist'].iloc[i])
                current_values['current_b'].append(outliers['intercept_dist'].iloc[i])

        # match the fixed index to the actual data index
        for i in current_values['current_index']:
            inf_points['index'].extend([fixed_index[i]])
            # fixed_index.pop(i)

        # save the results
        inf_points['influence'].extend(current_values['current_influence'])
        inf_points[f'{q}_quantile'].extend(current_values['current_quantile'])
        inf_points['res_alpha'].extend(current_values['current_alpha'])
        inf_points['res_b'].extend(current_values['current_b'])

        # remove outlier from data
        features = np.delete(features, current_values['current_index'], axis=0)
        label = np.delete(label, current_values['current_index'])

    # convert results to dataframa and return
    if inf_points['index'] == []:
        df = np.nan
    else:
        df = pd.DataFrame(inf_points)

    print(f"\n ** process finished in {loop} steps, {len(inf_points['index'])} outliers found **")

    return df
