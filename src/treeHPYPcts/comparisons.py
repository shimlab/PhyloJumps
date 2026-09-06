import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def aic_wt(x):
    delta_aic = x - min(x)
    return np.exp(-0.5 * delta_aic) / np.sum(np.exp(-0.5 * delta_aic))

def summarise_reml_results(folder):
    no_jump = ['BM', 'OU', 'EB']
    data = []
    for f in os.listdir(folder):
        if f.endswith('.json'):
            treeidx = f.replace('tree', '').replace('.json', '')
            with open(f"{folder}/{f}", 'r') as file:
                result = json.load(file)
                for k in result:
                    if 'AIC' in result[k]:
                        data.append({'tree':int(treeidx), 'model': k, 'aic': result[k]['AIC'][0]})
                    else:
                        data.append({'tree':int(treeidx), 'model' :k, 'aic':np.nan})
    data = pd.DataFrame(data)
    data = data[~data['aic'].isna()]
    data['model_has_jp'] = data['model'].map(lambda x: not x in no_jump)
    data['tree_has_jp'] = data['tree'] >= 100
    return data

def summarise_treehpyp_results(folder, ground_truth = None):
    data = []
    for f in os.listdir(folder):
        if f.endswith('summary.csv'):
            df = pd.read_csv(f"{folder}/{f}")
            split = f.split('_')
            model = split[1]
            tree_index = int(split[0])
            njumps = df['predicted_jp'].sum()
            has_jump = njumps > 1
            bayes_factor = df['bayes_factor'].iloc[0]
            post_probs = df['post_prob'][1:]
            post_prob_1jp = min(post_probs[post_probs > 1e-6]) if np.any(post_probs > 1e-6) else 0.
            row = {'tree':tree_index,'model':model, 'njumps':njumps, 'has_jump': has_jump, 'bayes_factor':bayes_factor, 'post_prob_1jp':post_prob_1jp}
            if ground_truth is not None:
                predicted_jp = df['predicted_jp'].tolist()
                found_jp = True
                for i,j in enumerate(ground_truth):
                    if j >= 1:
                        if predicted_jp[i] < 1: found_jp = False
                row['found_jp'] = found_jp
            data.append(row)


    data = pd.DataFrame(data)
    data['tree_has_jump'] = data['tree'] >= 100
    data['bayes_factor'] = data['bayes_factor'].replace(np.inf, 1e4)
    
    return data

def get_min_aics(df):
    jump = df[df.model_has_jp]
    no_jump = df[~ df.model_has_jp]
    aic_jump = jump.loc[jump.groupby('tree')['aic'].idxmin()]
    aic_njump = no_jump.loc[no_jump.groupby('tree')['aic'].idxmin()]
    return pd.concat([aic_jump, aic_njump])

def probs_roc(ground_truth, probs):
    thresholds = np.linspace(0, 1, 1000)
    tprs = []
    fprs = []
    for p in thresholds:
        predicted = probs > p
        tn,fp,fn,tp = confusion_matrix(ground_truth, predicted).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        tprs.append(tpr)
        fprs.append(fpr)

    return tprs, fprs, thresholds

def bayes_roc(ground_truth, bfs):
    bayes_noinf = sorted(bfs[bfs != np.inf])
    thresholds = []
    # get thresholds based off of evenly spaced values for bfs
    thresholds.extend(np.linspace(0, bayes_noinf[0], 5))
    for i in range(1, len(bayes_noinf)):
        x = np.linspace(bayes_noinf[i-1], bayes_noinf[i], 5)
        thresholds.extend(x)
    thresholds.append(max(bayes_noinf) + 1)
    tprs = []
    fprs = []
    for bf in thresholds:
        predicted = bfs > bf
        tn,fp,fn,tp = confusion_matrix(ground_truth, predicted).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        tprs.append(tpr)
        fprs.append(fpr)

def get_ratios(no_jp_aic, jp_aic):
    ratios = []
    alternative = []
    tree_has_jp = []
    null = []
    alt_model = []
    aicwt_1 = []
    aicwt_0 = []
    ratios_jp_vs_nojp = []
    for x,y in zip(no_jp_aic.iterrows(), jp_aic.iterrows()):
        row1 = x[1]
        row2 = y[1]
        # alternative is no jump
        if row1.aic < row2.aic:
            m_aic = row1.aic
            delta_aic = np.array([row1.aic, row2.aic]) - m_aic
            aic_wts = np.exp(-0.5*delta_aic) / np.sum(np.exp(-0.5*delta_aic))
            ratios.append(aic_wts[0]/ aic_wts[1])
            aicwt_0.append(aic_wts[1])
            aicwt_1.append(aic_wts[0])
            alternative.append(False)
            null.append(row2.model)
            alt_model.append(row1.model)
        else: #alternative is jump
            m_aic = row2.aic
            delta_aic = np.array([row1.aic, row2.aic]) - m_aic
            aic_wts = np.exp(-0.5*delta_aic) / np.sum(np.exp(-0.5*delta_aic))
            ratios.append(aic_wts[1] / aic_wts[0])
            aicwt_0.append(aic_wts[0])
            aicwt_1.append(aic_wts[1])
            alternative.append(True)
            null.append(row1.model)
            alt_model.append(row2.model)

        # always include jp vs no jp ratio
        d_aic = np.array([row1.aic, row2.aic]) - row2.aic
        aic_wt = np.exp(-0.5 * d_aic) / np.sum(np.exp(-0.5 * d_aic))
        ratios_jp_vs_nojp.append(aic_wt[1] / aic_wt[0])

        # doesn't matter which row is used
        tree_has_jp.append(row1.tree_has_jp)

    return pd.DataFrame({
        'null' : null, 'alt_model':alt_model,
        'aicwt_0':aicwt_0, 'aicwt_1': aicwt_1,
        'ratio':ratios, 'ratio_jp_vs_nojp' :ratios_jp_vs_nojp,
        'alternative': alternative,
        'tree_has_jp': tree_has_jp
    })

def get_aic_wts(aic_nojp, aic_jp):
    all_aics = np.column_stack((aic_nojp, aic_jp))
    min_aics = np.min(all_aics, axis = 1)
    delta_aics = all_aics - np.column_stack((min_aics, min_aics))
    aic_wts = np.exp(-0.5*delta_aics) / np.column_stack((np.sum(np.exp(-0.5*delta_aics), axis = 1),np.sum(np.exp(-0.5*delta_aics), axis = 1)))
    return aic_wts


def aic_ratio_roc(ratio_df, early_stop=True):
    thresholds = sorted(ratio_df.ratio.values)
    tprs = []
    fprs = []
    n = []
    for r in thresholds:
        significant = ratio_df[ratio_df.ratio > r]
        n.append(significant.shape[0])
        predicted = significant.alternative
        ground_truth = significant.tree_has_jp
        tp = 0
        fn = 0
        tn = 0
        fp = 0
        for p, t in zip(predicted, ground_truth):
            if p and not t:
                fp += 1
            if not p and t:
                fn += 1
            if p and t:
                tp += 1
            if not p and not t:
                tn += 1
        print("Threshold", r)
        print(
            np.array([[tp, fp], [fn, tn]])
        )
        try:
            tpr = tp / (tp + fn)
        except:
            break
            print('break here')
        try:
            fpr = fp / (tn + fp)
        except:
            break
            print('break here')
        tprs.append(tpr)
        fprs.append(fpr)

    return tprs, fprs, thresholds, n

def aic_roc(ground_truth, aic_nojump, aic_jump, remove_false = True):
    aic_nojump = np.array(aic_nojump)
    aic_jump = np.array(aic_jump)
    ratios = sorted(aic_jump/aic_nojump)
    thresholds = []
    # get thresholds based off of evenly spaced values for bfs
    thresholds.extend(np.linspace(0, ratios[0], 5))
    for i in range(1, len(ratios)):
        x = np.linspace(ratios[i-1], ratios[i], 5)
        thresholds.extend(x)
    thresholds.append(max(ratios) + 1)
    tprs = []
    fprs = []
    n = []
    ground_truth = np.array(ground_truth)
    aic_ratios = aic_jump / aic_nojump
    for ratio in thresholds:
        predicted = aic_ratios > ratio
        if remove_false:
            new_ground_truth = ground_truth[predicted]
            predicted = predicted[predicted]
            n.append(len(predicted))
            tn,fp,fn,tp = confusion_matrix(new_ground_truth, predicted).ravel()
        else:
            tn,fp,fn,tp = confusion_matrix(ground_truth, predicted).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        tprs.append(tpr)
        fprs.append(fpr)

    return tprs, fprs, thresholds, n

def aic_wt_model_select(reml_df):
    def second_largest(series):
        unique_sorted = series.unique()
        if len(unique_sorted) < 2:
            return None
        return sorted(unique_sorted, reverse=True)[1]

    for tree in reml_df['tree'].unique():
        reml_df.loc[(reml_df['tree'] == tree) & ~reml_df['model_has_jp'], 'keep'] = True
        jp_models = reml_df.loc[(reml_df['tree'] == tree) & reml_df['model_has_jp']]
        reml_df.loc[(reml_df['tree'] == tree) & reml_df['model_has_jp'], 'keep'] = (
                jp_models['aic'] == jp_models['aic'].min()
        )

    best_aics = get_min_aics(reml_df)
    best_aics['aic_wt'] = aic_wt(best_aics['aic'])
    jp_aics = best_aics.loc[best_aics['model_has_jp']].sort_values('tree')['aic_wt'].values
    nojp_aics = best_aics.loc[~best_aics['model_has_jp']].sort_values('tree')['aic_wt'].values


    reml_df['aic_wt'] = reml_df.groupby('tree')['aic'].transform(aic_wt)
    reml_df['max_aicwt'] = reml_df.groupby('tree')['aic_wt'].transform('max')
    reml_df['ratio_with_max'] = reml_df['max_aicwt'] / reml_df['aic_wt']
    reml_df['best_model'] = (reml_df['aic_wt'] == reml_df['max_aicwt'])
    second_best_aic = reml_df.groupby('tree')['aic_wt'].apply(second_largest)
    best_models = reml_df.loc[reml_df['best_model'], ['tree', 'model', 'model_has_jp', 'tree_has_jp', 'aic_wt']].reset_index(drop=True)
    best_models = best_models.sort_values('tree')
    best_models['second_best_aic'] = second_best_aic
    # get rid of duplicates/AIC ties?
    best_models = best_models.groupby('tree').head(1)
    pivot = reml_df.pivot_table(values = 'aic', index = 'tree', columns = 'model_has_jp', aggfunc='sum').reset_index()
    best_models['aic_ratio_jpnojp'] = jp_aics / nojp_aics
    return best_models