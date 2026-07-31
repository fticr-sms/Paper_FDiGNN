
### Needed Imports
import pandas as pd
import numpy as np

import scipy.stats as stats

import sklearn.ensemble as skensemble
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score, mean_squared_error
import sklearn.model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from fractions import Fraction

# Our Python package
import metabolinks.transformations as transf

import re
from tqdm import tqdm

# multianalysis.py file (has to be in the same folder)
from multianalysis import _generate_y_PLSDA, _calculate_vips

####### ---------------------------------- #######

# Furthermore, all the functions in step 5 and 6 are altered versions from the functions in multianalysis.py
# That were used for the BinSim paper.

# Finally, the function create_element_counts in step 8 was adapted from the Python package metabolinks and
# the function element_composition was also taken from metabolinks.

####### ---------------------------------- #######

### Step 1 Functions
### Functions related to Data Filtering

def basic_feat_filtering(file, target=None, filt_method='total_samples', filt_kw=2,
                  extra_filt=None, extra_filt_data=None):
    "Performs feature filtering in 2 steps."

    # Filtering based on the sampels each feature appears in
    if filt_method == 'total_samples': # Filter features based on the times (filt_kw) they appear in the dataset
        # Minimum can be a percentage if it is a value between 0 and 1!
        data_filt = transf.keep_atleast(file, minimum=filt_kw)
    elif filt_method == 'class_samples': # Features retained if they appear filt_kw times in the samples of at least one class
        # Minimum can be a percentage if it is a value between 0 and 1!
        data_filt = transf.keep_atleast(file, minimum=filt_kw, y=np.array(target))
    elif filt_method == None: # No filtering
        data_filt = file.copy()
    else:
        raise ValueError('Feature Filtering strategy not accepted/implemented in function. Implement if new strategy.')

    # Extra filtering based if the features are annotated
    if extra_filt == 'Formula': # Keep only features with a formula annotated on the dataset
        meta_cols_formulas = [i for i in extra_filt_data.columns if 'formulas' in i]
        if 'Formula' in extra_filt_data.columns:
            idxs_to_keep = [i for i in data_filt.columns if type(extra_filt_data.loc[i, 'Formula']) == str]
        else:
            idxs_to_keep = []
        for col in meta_cols_formulas:
            idxs_to_keep.extend([i for i in data_filt.columns if type(extra_filt_data.loc[i, col]) == list])
        idxs_to_keep = pd.unique(np.array(idxs_to_keep))
        data_filt = data_filt.loc[:,idxs_to_keep]

    elif extra_filt == 'Name': # Keep only features with a name annotated on the dataset
        meta_cols_names = [i for i in extra_filt_data.columns if 'names' in i]
        if 'Name' in extra_filt_data.columns:
            idxs_to_keep = [i for i in data_filt.columns if type(extra_filt_data.loc[i, 'Name']) == str]
        else:
            idxs_to_keep = []
        for col in meta_cols_names:
            idxs_to_keep.extend([i for i in data_filt.columns if type(extra_filt_data.loc[i, col]) == list])
        idxs_to_keep = pd.unique(np.array(idxs_to_keep))
        data_filt = data_filt.loc[:,idxs_to_keep]

    elif extra_filt == None: # No extra filtering
        data_filt = data_filt.copy()
    else:
        raise ValueError('Feature Filtering strategy not accepted/implemented in function. Implement if new strategy.')

    return data_filt


def min_intensity_filter(df, sample_cols, intensity_calculation='mean', threshold_type='% Based', threshold_value=0.1):
    "Filter features based on their intensity."

    # Calculating Intensities
    if intensity_calculation == 'mean':
        ints =  df[sample_cols].mean(axis=1)
    elif intensity_calculation == 'median':
        ints =  df[sample_cols].median(axis=1)

    # Thresholding
    if threshold_type == '% Based':
        # Remove the X % percent of features with the lowest intensities
        #ints = ints.sort_values()
        #n_feats_to_remove = int(len(df)*threshold_value)
        idxs_to_remove = ints[ints < ints.quantile(threshold_value)].index
        df = df.drop(index=idxs_to_remove)

    elif threshold_type == 'Intensity value':
        # Remove the features below a hard intensity threshold
        idxs_to_remove = ints[ints < threshold_value].index
        df = df.drop(index=idxs_to_remove)

    else:
        raise ValueError('Intensity-based Filtering strategy not accepted/implemented. Implement if new strategy.')

    return df


def QC_reproducibility_filter(df, qc_sample_cols, rsd_threshold=0.3):
    "Filter features based on the relative standard deviation of features in the Quality Control samples."

    # Only applicable when there are more than 2 QC samples
    if len(qc_sample_cols) > 2:
        # Select QC samples
        qc_samples = df[qc_sample_cols]

        # Calculate the relative standard deviation for each feature
        qc_rsd = qc_samples.std(axis=1)/qc_samples.mean(axis=1)
        # Remove indexes that are above a designated rsd_threshold
        idxs_to_remove = qc_rsd[(qc_rsd > rsd_threshold)].index
        df = df.drop(index=idxs_to_remove)

    return df


def variance_based_filter(df, sample_cols, variance_calculation_type='Relative Standard Deviation',
                          feat_to_remove_percent=0.1):
    "Filter features based on their variance (features with lower variance are removed)."

    # DataFrame with the samples
    sample_df = df[sample_cols]

    # Mean and Standard Deviation from the features
    mean_sample_df = sample_df.mean(axis=1)
    std_sample_df = sample_df.std(axis=1)

    # Calculate the varinces for the different possible types than can be chosen
    if variance_calculation_type == 'Inter-Quartile Range':
        iqr_sample_df = sample_df.quantile(0.75, axis=1) - sample_df.quantile(0.25, axis=1)
        var_df = iqr_sample_df
    elif variance_calculation_type == 'Standard Deviation':
        var_df = std_sample_df
    elif variance_calculation_type == 'Relative Standard Deviation':
        rsd_sample_df = std_sample_df / mean_sample_df
        var_df = rsd_sample_df
    elif variance_calculation_type == 'Median Absolute Deviation':
        mad_sample_df = (sample_df.T - mean_sample_df).abs().T.median(axis=1)
        var_df = mad_sample_df
    elif variance_calculation_type == 'Relative Median Absolute Deviation':
        mad_sample_df = (sample_df.T - mean_sample_df).abs().T.median(axis=1)
        rmad_sample_df = mad_sample_df / sample_df.median(axis=1)
        var_df = rmad_sample_df
    else:
        raise ValueError('Variance-based Filtering strategy not accepted/implemented. Implement if new strategy.')

    # Select idxs_to_keep
    idxs_to_keep = var_df[var_df >= var_df.quantile(feat_to_remove_percent)].index

    return df.loc[idxs_to_keep]


def filtering_data_metabolomics(df, sample_cols, qc_cols, target, # DataFrame, Columns of Samples, of QC samples and the target
                               # Filter 1: Filter based on the number of samples features appear in
                               basic_filt='total_samples', # 'total_samples', 'class_samples', None
                               n_min_samples_feature_appear=2, # Min. number of samples a feature must appear in data/class
                               # Filter 2: Intensity based filter
                               int_based_filter=False, # True or False whether you apply it
                               intensity_calculation='mean', # 'median'
                               threshold_type='Intensity value', # '% Based'
                               threshold_value=1*10**6, # Fraction such as 0.1 for % Based
                               # Filter 3: QC sample feature varation based filter
                               QC_filter=False, # True or False whether you apply it
                               rsd_threshold=0.25, # Fraction of features to remove in this check
                               # Filter 4: Sample feature variance based filter
                               var_based_filter=False, # True or False whether you apply it
                               variance_calculation_type='Relative Standard Deviation', # 'Inter-Quartile Range',
                                #'Standard Deviation', 'Relative Standard Deviation', 'Median Absolute Deviation',
                                #'Relative Median Absolute Deviation'
                               feat_to_remove_percent=0.10, # Fraction of features to remove in this check
                               # Select features to keep independent of the filtering performed
                               feats_to_keep=[]
                               ):
    """Performs up to 4 types of metabolic feature filtering from metabolomics datasets.

       return: Filtered DataFrame."""

    # Select Features to keep even if they are filtered
    rows_to_keep = df.loc[feats_to_keep]

    # Filter 1: Filter based on the number of samples features appear in
    if basic_filt != None:
        meta_cols = [i for i in df.columns if i not in sample_cols]
        temp = basic_feat_filtering(df[sample_cols].T, target=target,
                                    filt_method=basic_filt, # Method
                                    filt_kw=n_min_samples_feature_appear) # Make a sample have to appear
        df = pd.concat((df[meta_cols].reindex(temp.columns), temp.T), axis=1)

    # Filter 2: Intensity based filter
    if int_based_filter:
        df = min_intensity_filter(df, sample_cols, intensity_calculation=intensity_calculation,
                                threshold_type=threshold_type, threshold_value=threshold_value)

    # Filter 3: QC sample feature varation based filter
    if QC_filter:
        df = QC_reproducibility_filter(df, qc_cols, rsd_threshold=rsd_threshold)

    # Filter 4: Sample feature variance based filter
    if var_based_filter:
        df = variance_based_filter(df, sample_cols,
                                variance_calculation_type=variance_calculation_type,
                                feat_to_remove_percent=feat_to_remove_percent)

    # Re-add or keep features explicitly told to leave in the DataFrame
    for i in rows_to_keep.index:
        df.loc[i] = rows_to_keep.loc[i]

    return df




### Function to give an overall characterization of data

def characterize_data(dataset, name='dataset', target=None):
    "Returns some basic characteristics about the dataset."

    n_samples, n_feats = dataset.shape

    # Target related characteristics
    if target:
        n_classes = len(np.unique(np.array(target)))
        Samp_Class = len(target)/len(np.unique(np.array(target))) # Number of Sample per Class

    avg_feature_value = dataset.values.flatten()[~np.isnan(dataset.values.flatten())].mean() # Mean value in the dataset
    max_feature_value = dataset.values.flatten()[~np.isnan(dataset.values.flatten())].max() # Maximum value in the dataset
    min_feature_value = dataset.values.flatten()[~np.isnan(dataset.values.flatten())].min() # Minimum value in the dataset
    std_feature_value = dataset.values.flatten()[~np.isnan(dataset.values.flatten())].std() # Standard Deviation value
    median_feature_value = np.median(dataset.values.flatten()[~np.isnan(dataset.values.flatten())]) # Median value in the dataset

    if target:
        return {'Dataset': name,
                '# samples': n_samples,
                '# features': n_feats,
                'feature value average (std)': f'{avg_feature_value:.3f} ({std_feature_value:.3f})',
                'feature value ranges': f'({min_feature_value} - {max_feature_value})',
                'feature value median': median_feature_value,
                '# classes': n_classes,
                'samples / class': Samp_Class,
                }
    else:
        return {'Dataset': name,
                '# samples': n_samples,
                '# features': n_feats,
                'Feature value average (std)': f'{avg_feature_value:.3f} ({std_feature_value:.3f})',
                'Feature value ranges': f'({min_feature_value} - {max_feature_value})',
                'Feature value median': median_feature_value,
                }


### Step 1.2 Functions
### Functions related to metabolite annotations

def metabolite_annotation(annotated_data, dbs, ppm_margin, mass_val_col, adducts_to_consider={}, only_select_min_ppm=False):
    # Go through the selected databases
    for d in dbs:

        # Calculating adduct masses in annotation databases
        for adduct in adducts_to_consider:
            dbs[d]['DB'][adduct] = (dbs[d]['DB'][dbs[d]['Mass_col']] + Fraction(adducts_to_consider[adduct])).astype(float)

        print('Annotating with',d, end=' ')
        matched_ids_col = 'Matched '+d+' IDs'
        matched_names_col = 'Matched '+d+' names'
        matched_formulas_col = 'Matched '+d+' formulas'
        matched_add_col = 'Matched '+d+' adducts'
        match_count_col = d+' match count'
        annotated_data[matched_ids_col] = [[] for i in range(len(annotated_data))]
        annotated_data[matched_names_col] = [[] for i in range(len(annotated_data))]
        annotated_data[matched_formulas_col] = [[] for i in range(len(annotated_data))]
        annotated_data[matched_add_col] = [[] for i in range(len(annotated_data))]
        if d == 'HMDB':
            annotated_data['Matched KEGG IDs'] = [[] for i in range(len(annotated_data))]
        annotated_data[match_count_col] = ""

        # And for each metabolic feature
        for a in tqdm(annotated_data.index):
            # Information to store
            matched_ids = []
            matched_names = []
            matched_formulas = []
            matched_adds = []

            if d == 'HMDB':
                matched_keggs = []

            ppm_dev = pd.DataFrame()

            # Go through every adduct column in the database
            for ad_col in adducts_to_consider:
                mass_values = dbs[d]['DB'][ad_col]
                # Calculate ppm deviation and select compounds within the margin
                ppm_dev_ad = abs((mass_values-annotated_data[mass_val_col][a])/annotated_data[mass_val_col][a])*10**6
                ppm_dev_ad = ppm_dev_ad[ppm_dev_ad<ppm_margin] # ppm_margin used here
                # Join candidates to previous candidates
                ppm_dev = pd.concat((ppm_dev, ppm_dev_ad))
                matched_adds.extend([ad_col,] * len(ppm_dev_ad))

            # Reduce candidates to those which have the minimum ppm deviation
            if only_select_min_ppm:
                if len(ppm_dev) > 0:
                    min_ppm_dev = ppm_dev.min().values[0]
                    matched_adds = list(np.array(matched_adds)[ppm_dev[0].values == min_ppm_dev])
                    ppm_dev = ppm_dev[ppm_dev[0].values == min_ppm_dev]

            for i in ppm_dev.index:
                matched_ids.append(i)
                matched_names.append(dbs[d]['DB'][dbs[d]['Name_col']][i])
                matched_formulas.append(dbs[d]['DB'][dbs[d]['Formula_col']][i])
                if d == 'HMDB':
                    matched_keggs.append(dbs[d]['DB']['kegg'][i])

            # Perform annotation (if any compound was found)
            if len(matched_ids) > 0:
                annotated_data.at[a, matched_ids_col].extend(matched_ids)
                annotated_data.at[a, matched_names_col].extend(matched_names)
                annotated_data.at[a, matched_formulas_col].extend(matched_formulas)
                annotated_data.at[a, matched_add_col].extend(matched_adds)
                if d == 'HMDB':
                    annotated_data.at[a, 'Matched KEGG IDs'].extend(matched_keggs)
                annotated_data.at[a, match_count_col] = len(matched_ids)
            else:
                annotated_data.at[a, matched_ids_col] = np.nan
                annotated_data.at[a, matched_names_col] = np.nan
                annotated_data.at[a, matched_formulas_col] = np.nan
                annotated_data.at[a, matched_add_col] = np.nan
                if d == 'HMDB':
                    annotated_data.at[a, 'Matched KEGG IDs'] = np.nan
                annotated_data.at[a, match_count_col] = np.nan
        print(f'-> Annotated {annotated_data[matched_ids_col].notnull().sum()} compounds')
        print('---------------')
    return annotated_data

# If your database doesn't have the monoisotopic masses of the compounds, but has the formulas, this will calculate them:
chemdict = {'H':1.007825031898,
            'C':12.000000000,
            'N':14.00307400425,
            'O':15.99491461926,
            'Na':22.98976928195,
            'P':30.97376199768,
            'S':31.97207117354,
            'Cl':34.968852694,
            'F':18.99840316207,
            'I':126.904473,
            'Ca':39.96259092,
            'Mg':23.985041709,
            'K':38.963706493,
            'Cr':49.9460413,
            'Co':58.9331943,
            'Cu':62.9295973,
            'Fe':55.9349362,
            'Al':26.98153843,
            'Mo':97.9054041,
            'Rb':84.911789743,
            'Mn':54.9380432,
            'Se':79.9165226,
            'Zr':89.90469888,
            'Ga':68.9255738,
            'Te':129.906222758,
            'Br':78.9183387,
            'Sn':119.9022026,
            'Ti':47.94794098,
            'W':183.9509335,
            'Si':27.9769265353,
            'Bi':208.980401,
            'As':74.9215956,
            'B':11.009305178,
            'Be':9.01218315,
            'Ni':57.9353423,
            'Ge':75.92140271,
            'V':50.9439573,
            'Ag':106.905092,
            'Hg':201.9706445,
            'Cd':113.9033652,
            'Sr':87.905612264,
            'Sb':120.903812,
            'Au':196.9665704,
            'Ba':137.9052472,
            'Ta':180.948001,
            'Pb':207.9766528,
            'Li':7.016003443,
            'Y':88.9058382,
            'Ru':101.9043403,
            'Cs':132.905451966,
            'Pd':105.9034808,
            'Pt':194.9647943,
            'Ce':139.905451,
            'La':138.906362,
            'Nd':141.907731,
            'Re':186.9557525,
            'Tl':204.9744278,
            'Gd':157.9241128,
            'Zn':63.9291425,
            'Hf':179.946561,
            'Th':232.038051,
            'He':4.00260325454,
            'Ar':39.962383122,
            'Nb':92.906371,
            'Sm':151.9197398,
            'Cm':243.061389322,
            'Eu':152.9212379,
            'Lu':174.9407778,
            'Sc':44.959074,
            'Fr':221.01425,
            'Pr':140.907661,
            'Tb':158.9253547,
            'Dy':163.9291815,
            'Ho':164.9303295,
            'Er':165.9302998,
            'Tm':168.9342195,
            'Os':191.961482,
            'Ir':192.9629249,
            'Ra':224.020203,
            'Tc':97.907219,
            'Xe':131.904160,
            'In':114.903877,
            'Kr':83.911507,
            'Ne':19.992439,
            'Ac':227.027740,
            'Bk':247.070297} 

def calculate_monoisotopic_mass(formula):
    """Returns the monoisotopic mass of a given formula."""

    composition = element_composition(formula) # Transform string formulas into dict format

    mass = Fraction(0)

    for e in composition:
        mass = mass + Fraction(chemdict[e])*Fraction(composition[e])

    return mass


### Step 1.3 Functions
### Function related to merge duplicate (or more) annotations

def duplicate_disambiguator(annotated_data, sample_cols, mcid, mass_col, prev_an_form_cols, multiple_adds=True, verbose=False):
    """Attempts to remove duplicate (or more) annotations of peaks by merging them when possible.

       See explanation of procedure in explanation of Step 1.3.

       returns: annotated_data (pandas DataFrame with data after merging), 
                mergings_performed (dictionary with summary of mergings made),
                merging_situations (dictionary with counts of times each situation of merging was used),
                merge_description (dictionary with descriptions of mergings made),
                merge_problems (dictionary with descriptions of possible problems to merge)
    """

    # To store results
    merge_problems = {}
    merging_situations = {'Overwrite': 0, 'Merge same adducts':0, 'Merge different adducts':0, 'Problems': 0}
    mergings_performed = dict(zip(mcid, [0,]*len(mcid)))
    merge_description = {}
    n_merged_peaks = 0
    n_dropped_peaks = 0
    n_total_peaks = len(annotated_data.index)
    #merge_number = 0
    old_n_merged_peaks = -1
    n_loops = 1

    # Make iterations while the difference between the new current problems and the old_problems is not 0
    # That is when they stop being resolved
    while n_merged_peaks - old_n_merged_peaks > 0:
        old_n_merged_peaks = n_merged_peaks
        merge_problems = {}
        n_problems = 0
        # For the different databases used
        for col in mcid:
            new_idxs = {} # To save the idx where the merged peaks will be
            lost_idxs = [] # To save the idxs which will have to be removed
            # Adjust names if necessary
            col_id = col
            if col not in prev_an_form_cols:
                col = 'Matched '+col+' IDs'
            # See repeating annotations
            repeating_names = annotated_data[col].value_counts()[annotated_data[col].value_counts()>1].index

            for name in repeating_names: # For each repeating annotation
                # Grab the part of the dataframe with the repeats
                if col in prev_an_form_cols:
                    subset_df = annotated_data[annotated_data[col] == name]
                else:
                    subset_df = annotated_data.loc[[i for i in annotated_data[col].index if annotated_data.loc[i, col] == name]]

                # See if merging is possible and storing meta data
                merge=True
                saving_annotations = {} # To store the annotations to keep in the merged line
                for col_alt in mcid: # All other databases
                    col_alt_id = col_alt
                    if col_alt not in prev_an_form_cols:
                        col_alt = 'Matched '+col_alt+' IDs'
                    if col_alt == col:
                        saving_annotations[col_alt_id] = subset_df[col_alt].value_counts().index[0]
                    if col_alt != col:
                        a = []
                        # See if there are annotations in the other databases
                        subset_notnull = subset_df[col_alt][subset_df[col_alt].notnull()]
                        if len(subset_notnull) < 2:
                            if len(subset_notnull) == 1: # If there is only one, save it
                                saving_annotations[col_alt_id] = subset_notnull.value_counts().index[0]
                            continue
                            # If there is not, nothing to save
                        else:
                            n_annotations = len(subset_notnull.value_counts())
                            # If there is more than one but they are all the same, save it
                            if n_annotations == 1:
                                saving_annotations[col_alt_id] = subset_notnull.value_counts().index[0]
                                #print(subset_notnull.value_counts().index)
                                continue
                            # If there are multiple different annotations, PROBLEM
                            # Merging will not happen
                            else:
                                prob_idx = tuple(subset_df.index)
                                if prob_idx not in merge_problems.keys():
                                    merge_problems[prob_idx] = {'Nº of peaks':len(subset_df),
                                                                'Annotation': name,
                                                                'Col Id.': col_id,
                                                                'Poss. Reason': col_alt_id}
                                    n_problems += 1
                                merge=False

                # Starting the merging process if possible
                if merge:
                    n_masses = subset_df[mass_col] # Get the masses to match

                    # Get the idxs positions of the repeated annotations in the dataframe
                    idxs = [annotated_data.index.get_loc(subset_df.index[i]) for i in range(len(subset_df))]

                    min_idx = min(idxs) # Grab the minimum that will become the merged peak

                    idxs.remove(min_idx)
                    lost_idxs.extend(idxs) # Add the other to lost_idxs that will be removed when all this is over

                    overwrite = False #
                    adducts_summed = False

                    # Get the intensity values for the new merged line
                    if 'Matched '+col_id+' adducts' not in subset_df.columns: # If not made in our software, it can't see
                        new_line = subset_df[sample_cols].max()
                    else:
                        add_series = subset_df['Matched '+col_id+' adducts'].value_counts().index
                        if len(add_series) == 1:
                            new_line = subset_df[sample_cols].max()
                        else:
                            new_line = subset_df[sample_cols].copy()
                            dif_ads = []
                            for ad in add_series:
                                idxs = [i for i in subset_df.index if subset_df.loc[i, 'Matched '+col_id+' adducts'] == ad]
                                new_df = new_line.loc[idxs].max()
                                dif_ads.append(new_df)
                            new_line = pd.concat(dif_ads, axis=1).sum(axis=1).replace({0:np.nan})
                            adducts_summed = True
                    # For each annotation, see if the highest intensity values all come from one line
                    # Probably a better way to do this
                    for i in range(len(subset_df)):
                        if new_line.notnull().sum() - (new_line == subset_df[sample_cols].iloc[i]).sum() == 0:
                            # If yes, then situation 1: Overwrite is the correct option here
                            overwrite = True

                            # The new id (bucket label) will be the same as in the line with all the higher intensities
                            # Store it
                            keep_id = subset_df.index[i]
                            new_idxs[annotated_data.index[min_idx]] = keep_id

                            # The new line will be identical to that line in terms of intensity
                            temp_full_new_line = subset_df.iloc[i].copy()

                            # Filling the meta data of the new line with the data stored in saving annotations
                            idx_max = subset_df.loc[:,sample_cols].mean(axis=1).idxmax()
                            for key in saving_annotations:
                                #col_alt = key
                                if key in prev_an_form_cols:
                                    #subset_notnull = subset_df[subset_df[col_alt].notnull()]
                                    #idx_max = subset_notnull.loc[:,sample_cols].mean(axis=1).idxmax()
                                    #temp_full_new_line.loc[key] = saving_annotations[key]
                                    # Column is our Formula Assignment column
                                    if key + ' Adduct' in annotated_data.columns:
                                        #temp = annotated_data.loc[[
                                        #    i for i in annotated_data.index if annotated_data.loc[i, key] == saving_annotations[key]]]
                                        #temp_full_new_line[key + ' Adduct'] = subset_df.iloc[0][key + ' Adduct']
                                        temp_full_new_line[[key, key+' Adduct']] = subset_df.loc[idx_max, [key, key+' Adduct']]
                                    else:
                                        temp_full_new_line.loc[key] = subset_df.loc[idx_max, key]

                                else:
                                    #col_alt = 'Matched '+col_alt+' IDs'
                                    #subset_notnull = subset_df[subset_df[col_alt].notnull()]
                                    #idx_max = subset_notnull.loc[:,sample_cols].mean(axis=1).idxmax()
                                    # Grab the rest of the columns with meta_data: IDs, names, formulas and match count
                                    #temp = annotated_data.loc[[
                                    #    i for i in annotated_data.index if annotated_data.loc[i,
                                    #                'Matched '+key+' IDs'] == saving_annotations[key]]]

                                    rel_cols = ['Matched '+key+' IDs', 'Matched '+key+' names', 'Matched '+key+' formulas',
                                                'Matched '+key+' adducts', key+' match count']
                                    if key == 'HMDB':
                                        if 'Matched KEGG IDs' in annotated_data.columns:
                                            rel_cols.append('Matched KEGG IDs')
                                    #temp_full_new_line[rel_cols] = temp.iloc[0][rel_cols]
                                    temp_full_new_line[rel_cols] = subset_df.loc[idx_max, rel_cols]

                            merging_situations['Overwrite'] = merging_situations['Overwrite'] + 1
                            mergings_performed[col_id] = mergings_performed[col_id] + 1
                            merge_description[keep_id] = {'DB': col_id, 'Repeating annotation': name,
                                    'Nº merged peaks': len(subset_df), 'Merged peaks': list(subset_df.index), 'Situation': 'Overwrite'}

                            # Putting the merged line in the DataFrame
                            annotated_data.loc[annotated_data.index[min_idx]] = temp_full_new_line.copy()

                            if verbose:
                                if adducts_summed:
                                    print('Adducts were summed:', list(subset_df.index), 'to', keep_id)

                                if col_id in prev_an_form_cols:
                                    print(f'Merging {len(subset_df)} peaks (bucket labels: {list(subset_df.index)}) into one ', end='')
                                    print(f'({subset_df.index[i]}) by overwriting due to repeating annotations by previous ', end='')
                                    print(f'Annotation/Formula {col_id}: {name}.')
                                    print('----------')
                                else:
                                    print(f'Merging {len(subset_df)} peaks (bucket labels: {list(subset_df.index)}) into one ', end='')
                                    print(f'({subset_df.index[i]}) by overwriting due to repeating annotations by {col_id}: {name}.')
                                    print('----------')
                            continue

                    # If Situation 1, we end here
                    if overwrite:
                        continue

                    # If not Situation 1
                    temp_full_new_line = annotated_data.iloc[min_idx].copy()
                    # The new line will have the higher intensities for each sample
                    temp_full_new_line.loc[sample_cols] = new_line

                    # Filling the meta data of the new line with the data stored in saving annotations
                    idx_max = subset_df.loc[:,sample_cols].mean(axis=1).idxmax()
                    for key in saving_annotations:
                        #col_alt = key
                        if key in prev_an_form_cols:
                            #subset_notnull = subset_df[subset_df[col_alt].notnull()]
                            #idx_max = subset_notnull.loc[:,sample_cols].mean(axis=1).idxmax()
                            #temp_full_new_line.loc[key] = saving_annotations[key]
                            # Column is our Formula Assignment column
                            if key + ' Adduct' in annotated_data.columns:
                                #temp = annotated_data.loc[[
                                #    i for i in annotated_data.index if annotated_data.loc[i, key] == saving_annotations[key]]]
                                #temp_full_new_line[key + ' Adduct'] = subset_df.iloc[0][key + ' Adduct']
                                temp_full_new_line[[key, key+' Adduct']] = subset_df.loc[idx_max, [key, key+' Adduct']]
                            else:
                                temp_full_new_line.loc[key] = subset_df.loc[idx_max, key]

                        else:
                            #col_alt = 'Matched '+col_alt+' IDs'
                            #subset_notnull = subset_df[subset_df[col_alt].notnull()]
                            #idx_max = subset_notnull.loc[:,sample_cols].mean(axis=1).idxmax()
                            # Grab the rest of the columns with meta_data: IDs, names, formulas and match count
                            #temp = annotated_data.loc[[
                            #    i for i in annotated_data.index if annotated_data.loc[i,
                            #                'Matched '+key+' IDs'] == saving_annotations[key]]]
                            rel_cols = ['Matched '+key+' IDs', 'Matched '+key+' names', 'Matched '+key+' formulas',
                                        'Matched '+key+' adducts', key+' match count']
                            if key == 'HMDB':
                                if 'Matched KEGG IDs' in annotated_data.columns:
                                    rel_cols.append('Matched KEGG IDs')
                            #temp_full_new_line[rel_cols] = temp.iloc[0][rel_cols]
                            temp_full_new_line[rel_cols] = subset_df.loc[idx_max, rel_cols]

                    # All that's left is the bucket label, Neutral Mass and m/z
                    if multiple_adds:
                        # See if the distance between the maximum and minimum mass is low aka they come from the same adduct
                        # If yes, Situation 2: bucket label, Neutral Mass and m/z peak will be the weighted averages of all the
                        # possible peaks
                        if max(n_masses) - min(n_masses) < 0.5:
                            # Get the new bucket label
                            new_bucket = np.average(n_masses, weights=subset_df[sample_cols].mean(axis=1))
                            new_idxs[annotated_data.index[min_idx]] = new_bucket
                            # Get the Neutral Mass
                            temp_full_new_line[mass_col] = new_bucket

                            # Storing info
                            situation = 'merging'
                            merging_situations['Merge same adducts'] = merging_situations['Merge same adducts'] + 1
                            mergings_performed[col_id] = mergings_performed[col_id] + 1
                            merge_description[new_bucket] = {'DB': col_id, 'Repeating annotation': name,
                                    'Nº merged peaks': len(subset_df), 'Merged peaks': list(subset_df.index),
                                    'Situation': 'Merging same adducts'}

                        else:
                            # Get the new bucket label
                            argmax_idx = subset_df.loc[:,sample_cols].mean(axis=1).argmax()
                            new_bucket = subset_df.iloc[argmax_idx].name
                            new_idxs[annotated_data.index[min_idx]] = new_bucket
                            temp_full_new_line[mass_col] = subset_df.iloc[argmax_idx][mass_col]

                            # Storing info
                            situation = 'merging different adducts'
                            merging_situations['Merge different adducts'] = merging_situations['Merge different adducts'] + 1
                            mergings_performed[col_id] = mergings_performed[col_id] + 1
                            merge_description[new_bucket] = {'DB': col_id, 'Repeating annotation': name,
                                    'Nº merged peaks': len(subset_df), 'Merged peaks': list(subset_df.index),
                                    'Situation': 'Merging different adducts'}


                    else:
                        # Get the new bucket label
                        new_bucket = np.average(n_masses, weights=subset_df[sample_cols].mean(axis=1))
                        new_idxs[annotated_data.index[min_idx]] = new_bucket
                        temp_full_new_line[mass_col] = np.average(n_masses, weights=subset_df.loc[:,sample_cols].mean(axis=1))

                        # Storing info
                        situation = 'merging'
                        merging_situations['Merge same adducts'] = merging_situations['Merge same adducts'] + 1
                        mergings_performed[col_id] = mergings_performed[col_id] + 1
                        merge_description[new_bucket] = {'DB': col_id, 'Repeating annotation': name, 
                                    'Nº merged peaks': len(subset_df), 'Merged peaks': list(subset_df.index),
                                    'Situation': 'Merging same adducts (no multiple ad.)'}

                    # Putting the merged line in the DataFrame
                    annotated_data.loc[annotated_data.index[min_idx]] = temp_full_new_line.copy()

                    if adducts_summed:
                        print('Adducts were summed:', list(subset_df.index), 'to', new_bucket)

                    if verbose:
                        if col_id in prev_an_form_cols:
                            print(f'Merging {len(subset_df)} peaks (bucket labels: {list(subset_df.index)}) into one ', end='')
                            print(f'({new_bucket}) by {situation} due to repeating annotations by previous ', end='')
                            print(f'Annotation/Formula {col_id}: {name}.')
                            print('----------')
                        else:
                            print(f'Merging {len(subset_df)} peaks (bucket labels: {list(subset_df.index)}) into one ', end='')
                            print(f'({new_bucket}) by {situation} due to repeating annotations by {col_id}: {name}.')
                            print('----------')

            # Removing lost idxs peaks
            #print(len(lost_idxs), len(set(lost_idxs)))
            named_idxs = []
            for idx in lost_idxs:
                named_idxs.append(annotated_data.index[idx])
            for idx in named_idxs:
                annotated_data = annotated_data.drop(index=idx)
                n_dropped_peaks +=1

            # Assigning the new bucket labels
            for old_idx, new_idx in new_idxs.items():
                annotated_data = annotated_data.rename(index= {old_idx : new_idx})
                n_merged_peaks += 1
        print(f'Finished Loop Nº {n_loops}. Performed {n_merged_peaks - old_n_merged_peaks} merges.')
        n_loops += 1

    merging_situations['Problems'] = n_problems
    print('-----------')
    print(f'Nº of Max. Problems in Merging found (True value might be lower): {n_problems}')

    print('-----------')
    print('Nº of Mergings:            ', n_merged_peaks)
    print('Nº of Peaks merged:        ', n_merged_peaks + n_dropped_peaks)
    print('Nº of Peaks dropped:       ', n_dropped_peaks)
    print('Nº of Peaks before merging:', n_total_peaks)
    print('Nº of Peaks after merging: ', len(annotated_data.index))

    return annotated_data, mergings_performed, merging_situations, merge_description, merge_problems


def individually_merging(annotated_data, given_idxs, sample_cols, mass_col, mcid, prev_annotations_cols=[],
                         prev_formula_cols=[], multiple_adds=True):
    """Attempts to merge a set of peaks given in the DataFrame.

       This ignores if the Formula assigned between the peaks is different. If it is, it will become the Formula of the
        peak with the highest average intensity across the samples

       returns: annotated_data (pandas DataFrame with data after merging), 
                merge_description (dictionary with description of the merging)
    """

    # To store results
    merge_description = {}

    # Grab the part of the dataframe you want to merge
    subset_df = annotated_data.loc[given_idxs]

    prev_an_form_cols = prev_annotations_cols + prev_formula_cols

    # See if the idxs made contain something capable to search for merge
    possible_merging = False
    for col in mcid:
        # Adjust names if necessary
        col_id = col
        if col not in prev_an_form_cols:
            col = 'Matched '+col+' IDs'
        if len(annotated_data.loc[given_idxs][col].value_counts()) == 1:
            an = annotated_data.loc[given_idxs][col].value_counts().iloc[0]
            if an == len(given_idxs):
                annotation = an
                possible_merging = True
                col_id = col_id
                break

    if possible_merging:
        new_idxs = {} # To save the idx where the merged peaks will be
        lost_idxs = [] # To save the idxs which will have to be removed

        saving_annotations = {} # To store the annotations to keep in the merged line
        for col_alt in mcid: # All other databases
            col_alt_id = col_alt
            if col_alt not in prev_formula_cols:
                if col_alt not in prev_an_form_cols:
                    col_alt = 'Matched '+col_alt+' IDs'
                if col_alt == col:
                    saving_annotations[col_alt_id] = subset_df[col_alt].value_counts().index[0]
                if col_alt != col:
                    a = []
                    # See if there are annotations in the other databases
                    subset_notnull = subset_df[col_alt][subset_df[col_alt].notnull()]
                    n_annotations = len(subset_notnull.value_counts())
                    if n_annotations < 2:
                        if n_annotations == 1: # If there is only one, save it
                            saving_annotations[col_alt_id] = subset_notnull.value_counts().index[0]
                        continue
                        # If there is not, nothing to save
                    else:
                        # If there are multiple different annotations, PROBLEM
                        # Merging will not happen
                        print(f'Problem Merging. Database: {col_id}, Annotation: {annotation}', end='')
                        print(f', Nº of peaks: {len(subset_df)}. Reason: {col_alt_id} annotation.')
                        print('Peaks must not contain different annotations for a database other than the "Formula" columns.')
                        return

        n_masses = subset_df[mass_col] # Get the mas values

        # Get the idxs positions of the repeated annotations in the dataframe
        idxs = [annotated_data.index.get_loc(subset_df.index[i]) for i in range(len(subset_df))]

        min_idx = min(idxs) # Grab the minimum that will become the merged peak

        idxs.remove(min_idx)
        lost_idxs.extend(idxs) # Add the other to lost_idxs that will be removed when all this is over

        overwrite = False #

        #new_line = subset_df[sample_cols].max()
        # Get the intensity values for the new merged line
        if 'Matched '+col_id+' adducts' not in subset_df.columns: # If not made in our software, it can't see
            new_line = subset_df[sample_cols].max()
        else:
            add_series = subset_df['Matched '+col_id+' adducts'].value_counts().index
            if len(add_series) == 1:
                new_line = subset_df[sample_cols].max()
            else:
                new_line = subset_df[sample_cols].copy()
                dif_ads = []
                for ad in add_series:
                    idxs = [i for i in subset_df.index if subset_df.loc[i, 'Matched '+col_id+' adducts'] == ad]
                    new_df = new_line.loc[idxs].max()
                    dif_ads.append(new_df)
                new_line = pd.concat(dif_ads, axis=1).sum(axis=1).replace({0:np.nan})
        # For each annotation, see if the highest intensity values all come from one line
        # Probably a better way to do this
        for i in range(len(subset_df)):
            if new_line.notnull().sum() - (new_line == subset_df[sample_cols].iloc[i]).sum() == 0:
                # If yes, then situation 1: Overwrite is the correct option here
                overwrite = True

                # The new id (bucket label) will be the same as in the line with all the higher intensities
                # Store it
                keep_id = subset_df.index[i]
                new_idxs[annotated_data.index[min_idx]] = keep_id

                # The new line will be identical to that line in terms of intensity
                # Formula becomes the same as in the peak with the highest intensity for all the samples
                temp_full_new_line = subset_df.iloc[i].copy()

                # Filling the meta data of the new line with the data stored in saving annotations
                idx_max = subset_df.loc[:,sample_cols].mean(axis=1).idxmax()
                for key in saving_annotations:
                    #col_alt = key
                    if key in prev_an_form_cols:
                        #subset_notnull = subset_df[subset_df[col_alt].notnull()]
                        #idx_max = subset_notnull.loc[:,sample_cols].mean(axis=1).idxmax()
                        #temp_full_new_line.loc[key] = saving_annotations[key]
                        # Column is our Formula Assignment column
                        if key + ' Adduct' in annotated_data.columns:
                            #temp = annotated_data.loc[[
                            #    i for i in annotated_data.index if annotated_data.loc[i, key] == saving_annotations[key]]].copy()
                            #temp_full_new_line[key + ' Adduct'] = subset_df.iloc[0][key + ' Adduct']
                            temp_full_new_line[[key, key+' Adduct']] = subset_df.loc[idx_max, [key, key+' Adduct']]
                        else:
                            temp_full_new_line.loc[key] = subset_df.loc[idx_max, key]

                    else:
                        #col_alt = 'Matched '+col_alt+' IDs'
                        #subset_notnull = subset_df[subset_df[col_alt].notnull()]
                        #idx_max = subset_notnull.loc[:,sample_cols].mean(axis=1).idxmax()
                        # Grab the rest of the columns with meta_data: IDs, names, formulas and match count
                        #temp = annotated_data.loc[[
                        #    i for i in annotated_data.index if annotated_data.loc[i,
                        #                'Matched '+key+' IDs'] == saving_annotations[key]]]

                        rel_cols = ['Matched '+key+' IDs', 'Matched '+key+' names', 'Matched '+key+' formulas',
                                    'Matched '+key+' adducts', key+' match count']
                        if key == 'HMDB':
                            if 'Matched KEGG IDs' in annotated_data.columns:
                                rel_cols.append('Matched KEGG IDs')
                        #temp_full_new_line[rel_cols] = temp.iloc[0][rel_cols]
                        temp_full_new_line[rel_cols] = subset_df.loc[idx_max, rel_cols]

                merge_description[keep_id] = {'DB': col_id, 'Repeating annotation': annotation,
                        'Nº merged peaks': len(subset_df), 'Merged peaks': list(subset_df.index),
                                            'Situation': 'Individual - Overwrite'}

                # Putting the merged line in the DataFrame
                annotated_data.loc[annotated_data.index[min_idx]] = temp_full_new_line.copy()

                if col_id in prev_an_form_cols:
                    print(f'Merging {len(subset_df)} peaks (bucket labels: {list(subset_df.index)}) into one ', end='')
                    print(f'({subset_df.index[i]}) by overwriting due to repeating annotations by previous ', end='')
                    print(f'Annotation/Formula {col_id}: {annotation}.')
                    print('----------')
                else:
                    print(f'Merging {len(subset_df)} peaks (bucket labels: {list(subset_df.index)}) into one ', end='')
                    print(f'({subset_df.index[i]}) by overwriting due to repeating annotations by {col_id}: {annotation}.')
                    print('----------')
                continue

        # If not Situation 1
        if not overwrite:

            # If not Situation 1
            temp_full_new_line = annotated_data.iloc[min_idx].copy()
            # The new line will have the higher intensities for each sample
            temp_full_new_line.loc[sample_cols] = new_line

            # Filling the meta data of the new line with the data stored in saving annotations
            idx_max = subset_df.loc[:,sample_cols].mean(axis=1).idxmax()
            for key in saving_annotations:
                #col_alt = key
                if key in prev_an_form_cols:
                    #subset_notnull = subset_df[subset_df[col_alt].notnull()]
                    #idx_max = subset_notnull.loc[:,sample_cols].mean(axis=1).idxmax()
                    #temp_full_new_line.loc[key] = saving_annotations[key]
                    # Column is our Formula Assignment column
                    if key + ' Adduct' in annotated_data.columns:
                        #temp = annotated_data.loc[[
                        #    i for i in annotated_data.index if annotated_data.loc[i, key] == saving_annotations[key]]].copy()
                        #temp_full_new_line[key + ' Adduct'] = subset_df.iloc[0][key + ' Adduct']
                        temp_full_new_line[[key, key+' Adduct']] = subset_df.loc[idx_max, [key, key+' Adduct']]
                    else:
                        temp_full_new_line.loc[key] = subset_df.loc[idx_max, key]

                else:
                    #col_alt = 'Matched '+col_alt+' IDs'
                    #subset_notnull = subset_df[subset_df[col_alt].notnull()]
                    #idx_max = subset_notnull.loc[:,sample_cols].mean(axis=1).idxmax()
                    # Grab the rest of the columns with meta_data: IDs, names, formulas and match count
                    #temp = annotated_data.loc[[
                    #    i for i in annotated_data.index if annotated_data.loc[i,
                    #                'Matched '+key+' IDs'] == saving_annotations[key]]]
                    rel_cols = ['Matched '+key+' IDs', 'Matched '+key+' names', 'Matched '+key+' formulas',
                                'Matched '+key+' adducts', key+' match count']
                    if key == 'HMDB':
                        if 'Matched KEGG IDs' in annotated_data.columns:
                            rel_cols.append('Matched KEGG IDs')
                    #temp_full_new_line[rel_cols] = temp.iloc[0][rel_cols]
                    temp_full_new_line[rel_cols] = subset_df.loc[idx_max, rel_cols]

            # Formula annotations becomes the ones from the peak with the highest average intensity across the samples
            for form in prev_formula_cols:
                if form in annotated_data.columns:
                    idx_max = subset_df.loc[:,sample_cols].mean(axis=1).idxmax()
                    #temp_full_new_line[form] = subset_df.iloc[argmax_idx][form]
                    if form + ' Adduct' in annotated_data.columns:
                        temp_full_new_line[[form, form+' Adduct']] = subset_df.loc[idx_max, [form, form+' Adduct']]
                    else:
                        temp_full_new_line.loc[form] = subset_df.loc[idx_max, form]

            # All that's left is the bucket label, Neutral Mass and m/z
            if multiple_adds:
                # If there are multiple adducts being used, see if the distance between the maximum and minimum mass is low
                # If yes, Situation 2: bucket label, Neutral Mass or m/z peak will be the weighted averages of all the
                # possible peaks
                if max(n_masses) - min(n_masses) < 0.5:
                    # Get the new bucket label
                    new_bucket = np.average(n_masses, weights=subset_df[sample_cols].mean(axis=1))
                    new_idxs[annotated_data.index[min_idx]] = new_bucket
                    temp_full_new_line[mass_col] = new_bucket

                    # Storing info
                    situation = 'merging'
                    merge_description[new_bucket] = {'DB': col_id, 'Repeating annotation': annotation,
                            'Nº merged peaks': len(subset_df), 'Merged peaks': list(subset_df.index),
                            'Situation': 'Individual - Merging same adducts'}

                else:
                    # Get the new bucket label
                    argmax_idx = subset_df.loc[:,sample_cols].mean(axis=1).argmax()
                    new_bucket = subset_df.iloc[argmax_idx].name
                    new_idxs[annotated_data.index[min_idx]] = new_bucket
                    temp_full_new_line[mass_col] = subset_df.iloc[argmax_idx][mass_col]

                    # Storing info
                    situation = 'merging different adducts'
                    merge_description[new_bucket] = {'DB': col_id, 'Repeating annotation': annotation,
                            'Nº merged peaks': len(subset_df), 'Merged peaks': list(subset_df.index),
                            'Situation': 'Individual - Merging different adducts'}


            else:
                # Get the new bucket label
                new_bucket = np.average(n_masses, weights=subset_df[sample_cols].mean(axis=1))
                new_idxs[annotated_data.index[min_idx]] = new_bucket
                temp_full_new_line[mass_col] = new_bucket

                # Storing info
                situation = 'merging'
                merge_description[new_bucket] = {'DB': col_id, 'Repeating annotation': annotation, 
                            'Nº merged peaks': len(subset_df), 'Merged peaks': list(subset_df.index),
                            'Situation': 'Individual - Merging same adducts (no multiple ad.)'}

            # Putting the merged line in the DataFrame
            annotated_data.loc[annotated_data.index[min_idx]] = temp_full_new_line.copy()

            if col_id in prev_an_form_cols:
                print(f'Merging {len(subset_df)} peaks (bucket labels: {list(subset_df.index)}) into one ', end='')
                print(f'({new_bucket}) by {situation} due to repeating annotations by previous ', end='')
                print(f'Annotations/Formula {col_id}: {annotation}.')
                print('----------')
            else:
                print(f'Merging {len(subset_df)} peaks (bucket labels: {list(subset_df.index)}) into one ', end='')
                print(f'({new_bucket}) by {situation} due to repeating annotations by {col_id}: {annotation}.')
                print('----------')

        # Removing lost idxs peaks
        named_idxs = []
        for idx in lost_idxs:
            named_idxs.append(annotated_data.index[idx])
        for idx in named_idxs:
            annotated_data = annotated_data.drop(index=idx)

        # Assigning the new bucket labels
        for old_idx, new_idx in new_idxs.items():
            annotated_data = annotated_data.rename(index= {old_idx : new_idx})

        return annotated_data, merge_description

    else:
        print(f'Merging was not possible since no identical annotation for peaks {given_idxs} was found.')


### Step 2 Functions
### Functions related to Data Pre-treatment

def missing_value_imputer(data_filt, mvi='min_sample', mvi_kw=1/5):
    "Performs Missing Value Imputation of choice based on parameters passed."

    # Missing Value Imputation
    if mvi == 'min_sample': # Replace NaN's by a fraction (mvi_kw) of the minimum value of the sample the NaN belongs to.
        imputed = transf.fillna_frac_min_feature(data_filt.T, fraction=mvi_kw).T
    elif mvi == 'min_feat': # Replace NaN's by a fraction (mvi_kw) of the minimum value of the feature the NaN belongs to.
        imputed = transf.fillna_frac_min_feature(data_filt, fraction=mvi_kw)
    elif mvi == 'min_data': # Replace NaN's by a fraction (mvi_kw) of the minimum value in the dataset.
        imputed = transf.fillna_frac_min(data_filt, fraction=mvi_kw)
    elif mvi == 'zero': # Replace NaN's by zero (no mvi_kw).
        imputed = transf.fillna_zero(data_filt)
    else:
        raise ValueError('Missing Value Imputation strategy not accepted/implemented in function. Implement if new strategy.')

    return imputed

def normalizer(data, norm='ref_feat', norm_kw='555.2692975341 Da'):
    "Performs Normalization of choice based on parameters passed."

    # Normalizations
    if norm == 'ref_feat': # Normalization by a reference feature indicated by the norm_kw
        N = transf.normalize_ref_feature(data, feature=norm_kw, remove=True)
    elif norm == 'total_sum': # Normalization by the total sum of intensities (no norm_kw)
        N = transf.normalize_sum(data)
    elif norm == 'PQN': # Normalization by Probabilistic Quotient Normalization (norm_kw is ref_sample, usually, 'mean')
        N = transf.normalize_PQN(data, ref_sample=norm_kw)
    elif norm == 'Quantile': # Normalization by Quantile Normalization (norm_kw is ref_feat, usually, 'mean')
        N = transf.normalize_quantile(data, ref_type=norm_kw)
    elif norm == None: # No Normalization
        N = data.copy()
    else:
        raise ValueError('Normalization strategy not accepted/implemented in function. Implement if new strategy.')

    return N

def transformer(N, tf='glog', tf_kw=None):
    "Performs Transformation of choice based on parameters passed."

    # Transformations
    if tf == 'glog': # Generalized Logarithmic Transformation with lambda = trans_kw, usually, None.
        NG = transf.glog(N, lamb=tf_kw)
    elif tf == None: # No Transformation
        NG = N.copy()
    else:
        raise ValueError('Transforamtion strategy not accepted/implemented in function. Implement if new strategy.')

    return NG

def scaler(NG, scaling='pareto', scaling_kw=None):
    "Performs Scaling of choice based on parameters passed."

    # Scalings
    if scaling == 'pareto': # Pareto scaling (no scaling_kw)
        NGP = transf.pareto_scale(NG)
    elif scaling == 'mean_center': # Just mean centering, no scaling (no scaling_kw)
        NGP = transf.mean_center(NG)
    elif scaling == 'auto': # Auto or Standard Scaling (no scaling_kw)
        NGP = transf.auto_scale(NG)
    elif scaling == 'range': # Range Scaling (no scaling_kw)
        NGP = transf.range_scale(NG)
    elif scaling == 'vast': # Vast Scaling (no scaling_kw)
        NGP = transf.vast_scale(NG)
    elif scaling == 'level': # Level Scaling (scaling_kw is boolean - True or False if average, usually False)
        NGP = transf.level_scale(NG, average=scaling_kw)
    elif scaling == None: # No Scaling
        NGP = NG.copy()
    else:
        raise ValueError('Scaling strategy not accepted/implemented in function. Implement if new strategy.')

    return NGP

def filtering_pretreatment(data, target, sample_cols,
                  filt_method='total_samples', filt_kw=2, # Filtering based on number of times features appear
                  extra_filt=None, # Filtering based on annotation of features ('Formula' or 'Name')
                  mvi='min_sample', mvi_kw=1/5, # Missing value imputation
                  norm='ref_feat', norm_kw='555.2692975341 Da', # Normalization
                  tf='glog', tf_kw=None, # Transformation
                  scaling='pareto', scaling_kw=None): # Scaling
    """Performs all feature filtering and data pre-treatments of choice based on parameters passed.

       Returns: Five DataFrames."""

    # Cols for the meta data and with the samples
    sample_cols = sample_cols
    meta_cols = [i for i in data.columns if i not in sample_cols]

    # Separates feature intensity data from "metadata" (m/z and annotations)
    meta_data = data[meta_cols]
    sample_data = data[sample_cols].T

    # Filtering
    filt_sample_data = basic_feat_filtering(sample_data, target, filt_method=filt_method, filt_kw=filt_kw,
                                            extra_filt=extra_filt, extra_filt_data=meta_data)

    # Treated data DataFrame
    imputed = missing_value_imputer(filt_sample_data.copy(), mvi=mvi, mvi_kw=mvi_kw) # Missing Value Imputation
    N = normalizer(imputed, norm=norm, norm_kw=norm_kw) # Normalization
    NG = transformer(N, tf=tf, tf_kw=tf_kw) # Transformation
    NGP = scaler(NG, scaling=scaling, scaling_kw=scaling_kw) # Scaling

    # Meta data DataFrame
    meta_data = meta_data.reindex(NGP.columns)

    # processed_data DataFrame
    norm_sample_data=normalizer(filt_sample_data, norm=norm, norm_kw=norm_kw) # Normalization without missing value imputation
    inverted_norm_sample_data = norm_sample_data.T
    processed_data = pd.concat([meta_data, inverted_norm_sample_data], axis=1,join='inner') # Obtaining the processed dataframe

    # univariate_data DataFrame
    univariate_data = N.copy()

    # BinSim DataFrame
    BinSim = filt_sample_data.mask(filt_sample_data.notnull(), 1).mask(filt_sample_data.isnull(), 0)
    if norm == 'ref_feat':
        BinSim = BinSim.drop(columns=norm_kw)

    return NGP, processed_data, univariate_data, meta_data, BinSim


### Step 3 Functions
### Functions related to Common and Exclusive Metabolites

def common(samples):
    """Given a list of n samples, compute common features (intersection).

       Returns a DataFrame with common features"""

    df = pd.concat(samples, axis=1, join='inner', keys=range(len(samples)))
    return df[0]

def exclusive(samples):
    """Given a list of samples, compute exclusive features for each sample.

       Returns a list of DataFrames with exclusive features for each corresponding sample in input"""

    # concat all samples
    concatenation = pd.concat(samples)

    # find indexes that occur only once
    reps = concatenation.index.value_counts()
    exclusive_feature_counts = reps[reps == 1]

    # keep only those in each sample
    exclusive = [s[s.index.isin(exclusive_feature_counts.index)] for s in samples]
    return exclusive


### Step 4 Functions
### Functions related to perform PCA (unsupervised analysis)

# Specific PCA calculation that also returns loadings
def compute_df_with_PCs_VE_loadings(df, n_components=5, whiten=True, labels=None, return_var_ratios_and_loadings=False):
    "Computes PCA as well as the variance explained in each component (PC) and associated loadings."

    # Fit PCA model and extract PCA scores, loadings and variance explained
    pca = PCA(n_components=n_components, svd_solver='full', whiten=whiten)
    pc_coords = pca.fit_transform(df)
    var_explained = pca.explained_variance_ratio_[:pca.n_components_]
    loadings = pca.components_[:pca.n_components_].T

    # concat labels to PCA coords (in a DataFrame)
    principaldf = pd.DataFrame(pc_coords, index=df.index, columns=[f'PC {i}' for i in range(1, pca.n_components_+1)])
    if labels is not None:
        labels_col = pd.DataFrame(labels, index=principaldf.index, columns=['Label'])
        principaldf = pd.concat([principaldf, labels_col], axis=1)

    # Return
    if not return_var_ratios_and_loadings:
        return principaldf
    else:
        return principaldf, var_explained, loadings


### Step 5 Functions
### Functions related to perform and evaluate through different metrics Random Forests and PLS-DA

### These functions are altered versions from the ones available in the multianalysis.py file from the BinSim paper

def RF_model(df, y, regres=False, return_cv=True, iter_num=1, n_trees=200, cv=None, n_fold=5, random_state=None,
             metrics = ('accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'), **kwargs):
    "Fitting RF models and rturning the models and their cross-validation scores."
    results = {}

    # Fit RF model section
    # Classifier or Regression model
    if regres:
        fitted_model = skensemble.RandomForestRegressor(n_estimators=n_trees)
    else:
        fitted_model = skensemble.RandomForestClassifier(n_estimators=n_trees)

    fitted_model = fitted_model.fit(df, y)
    results['model'] = fitted_model

    # Setting up variables for imp_feat storing
    imp_feat = np.zeros((iter_num * n_fold, len(df.columns)))
    f = 0

    if not return_cv:
        return(fitted_model)

    base_cv = cv

    store_res = {m:[] for m in metrics}

    # Go through the iterations specified and evaluate the model through cross-validation
    # Manually fit the model to train folds and extract feature importance information
    for itr in range(iter_num):
        # Evaluate RF model through cross-validation section (includes feature importance retrieval)
        cv = base_cv
        if cv is None:
            if random_state != None:
                cv = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True, random_state=random_state*(itr+1))
            else:
                cv = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True)

        if regres:
            rf = skensemble.RandomForestRegressor(n_estimators=n_trees)
        else:
            rf = skensemble.RandomForestClassifier(n_estimators=n_trees)

        # Evaluate model performance
        cv_res = sklearn.model_selection.cross_validate(rf, df, y, cv=cv, scoring=metrics, **kwargs)
        for i in metrics:
            store_res[i].extend(cv_res['test_'+i])

        # Split the data into folds and extract feature importance for each
        for train_index, test_index in cv.split(df, y):
            # Random Forest setup and fit
            if regres:
                rf = skensemble.RandomForestRegressor(n_estimators=n_trees)
            else:
                rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
            X_train, X_test = df.iloc[train_index, :], df.iloc[test_index, :]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            rf.fit(X_train, y_train)

            # Compute important features
            imp_feat[f, :] = rf.feature_importances_ # Importance of each feature
            f = f + 1

    # Collect and order all important features values from each Random Forest
    imp_feat_sum = imp_feat.sum(axis=0) / (iter_num * n_fold)
    results['imp_feat'] = sorted(enumerate(imp_feat_sum), key=lambda x: x[1], reverse=True)

    results.update(store_res)
    return results#{'model': fitted_model, 'cv_scores': scores}

def RF_ROC_cv(treated_data, target, pos_label, regres=False, n_trees=200, n_iter=1, cv=None, n_fold=5, random_state=None):
    """Fits and extracts Random Forest model data and calculates metrics to plot a ROC curve."""

    # Run classifier with cross-validation and plot ROC curves
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    base_cv = cv

    # Number of times Random Forest cross-validation is made
    # with `n_fold` randomly generated folds.
    for itr in range(n_iter):
        # Evaluate RF model through cross-validation section
        cv = base_cv
        if cv is None:
            if random_state != None:
                cv = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True, random_state=random_state*(itr+1))
            else:
                cv = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True)

        # Fit and evaluate a Random Forest model for each fold in cross validation
        for train_index, test_index in cv.split(treated_data, target):
            # Random Forest setup and fit
            if regres:
                rf = skensemble.RandomForestRegressor(n_estimators=n_trees)
            else:
                rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
            X_train, X_test = treated_data.iloc[train_index, :], treated_data.iloc[test_index, :]
            y_train, y_test = [target[i] for i in train_index], [target[i] for i in test_index]
            # Classifier fit
            rf.fit(X_train, y_train)

            # Metrics for ROC curve plotting
            scores = rf.predict_proba(X_test)[:,rf.classes_ == pos_label]

            fpr, tpr, _ = roc_curve(y_test, scores, pos_label=pos_label)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            #interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_score(y_test, scores))

    # Mean of every fold of the cross-validation
    mean_tpr = np.mean(tprs, axis=0)
    #mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # fpr - false positive rate, tpr - true positive rate, AUC - area under curve
    return {'average fpr': mean_fpr, 'average tpr': mean_tpr, 
            'upper tpr': tprs_upper, 'lower trp': tprs_lower,
            'mean AUC': mean_auc, 'std AUC': std_auc}


def permutation_RF(df, labels, regres=False, iter_num=100, n_trees=200, cv=None, n_fold=3, random_state=None,
                   metric = ('accuracy')):
    """Performs permutation test n times of a dataset for Random Forest classifiers giving its performance (estimated by
        cross-validation) for the original and all permutations made and respective p-value.

       df: Pandas DataFrame.
       labels: target labels.
       regres: bool (default: False); True when the biological problem is a regression and False when it is a classification.
       iter_num: int (default - 100); number of permutations made.
       n_trees: int (default - 200); number of trees in each Random Forest.
       cv: splitter class of sklearn.model_selection (default - None); choose a cross-validation method (and respective args); if
        None, the default method is stratified cross-validation.
       n_fold: int (default - 3); number of groups to divide dataset in for k-fold cross-validation (max n_fold = minimum number of
        samples belonging to one group) if cv is None.
       random_state: int (default - None); random seed given to make the permutations rng class labels.
       metric: tuple (default - ('accuracy')); metric to give to scikit-learn cross_validate.

       Returns: (scalar, list of scalars, scalar);
        estimated predictive accuracy of the non-permuted Random Forest model
        estimated predictive accuracy of all permuted Random Forest models
        p-value ((number of permutations with accuracy > original accuracy) + 1)/(number of permutations + 1).
    """

    # get a bit generator
    rng = np.random.default_rng(seed=random_state)

    # Setting up variables for result storing
    Perm = []
    # List of columns to shuffle and dataframe of the data to put columns in NewC shuffled order
    NewC = np.arange(df.shape[0])
    df = df.copy()

    # For dividing the dataset in balanced n_fold groups with a random random state maintained in all permutations (identical splits)
    if cv is None:
        cv = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True)


    for _ in range(iter_num + 1):
        # Number of different permutations + original dataset where Random Forest cross-validation will be made
        # Temporary dataframe with columns in order of the NewC
        temp = df.iloc[NewC, :]

        # Random Forest setup and cross-validation
        if regres:
            rf = skensemble.RandomForestRegressor(n_estimators=n_trees)
        else:
            rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
        cv_res = sklearn.model_selection.cross_validate(rf, temp, labels, cv=cv, scoring=metric)

        # Shuffle dataset columns - 1 permutation of the columns (leads to permutation of labels)
        rng.shuffle(NewC)
        # Appending K-fold cross-validation predictive accuracy
        Perm.append(np.mean(cv_res['test_score']))

    # Taking out K-fold cross-validation accuracy for the non-shuffled (labels) dataset and p-value calculation
    CV = Perm[0] # Non-permuted dataset results - Perm [0]
    pvalue = (sum(Perm[1:] >= Perm[0]) + 1) / (iter_num + 1)

    return CV, Perm[1:], pvalue


def optim_PLSDA_n_components(df, labels, regres=False, encode2as1vector=True, max_comp=15, min_comp=1, kf=None, n_fold=5, scale=False):
    """Searches for an optimum number of components to use in PLS-DA.

       df: DataFrame; X equivalent in PLS-DA (training vectors).
       labels: labels to target
       regres: bool (default: False); True when the biological problem is a regression and False when it is a classification.
       encode2as1vector: bool (default: True); when you have two classes encode them in a vector.
       max_comp: integer (default: 15); upper limit for the number of components used.
       min_comp: integer (default: 1); lower limit for the number of components used.
       kf: default None; pass a specific cross validation method from 
        https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators (3.1.2)
       n_fold: int (default: 5); number of groups to divide dataset in (max = min. number of samples belonging to one group)
       scale: bool (default: False); if data is scaled when inputted to PLS model (only true if scaling was not done earlier)

       Returns: (list, list), n-fold cross-validation (q2) score and r2 score for all components searched.
    """

    # Preparating lists to store results
    CVs = []
    CVr2s = []

    unique_labels = list(pd.unique(np.array(labels)))

    is1vector = (len(unique_labels) == 2 and encode2as1vector) or regres

    if regres:
        matrix = np.array(labels)
    else:
        matrix = _generate_y_PLSDA(labels, unique_labels, is1vector)


    if is1vector:
        # keep a copy to use later
        target1D = matrix.copy()

    # Repeating for each component from 1 to max_comp
    for i in range(min_comp, max_comp + 1):
        cv = []
        cvr2 = []

        if kf is None:
            kf = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True)

        # Splitting data into groups for n-fold cross-validation
        for train_index, test_index in kf.split(df, labels):
            # NOTE: scale=True if scaling has not been done up until this point
            plsda = PLSRegression(n_components=i, scale=scale)
            X_train, X_test = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()
            if not is1vector:
                y_train = matrix.iloc[train_index, :].copy()
                y_test = matrix.iloc[test_index, :].copy()
            else:
                y_train, y_test = target1D[train_index], target1D[test_index]
                correct = target1D[test_index]

            # Fitting the model
            plsda.fit(X=X_train, Y=y_train)

            # Obtain results with the test group
            #y_pred = plsda.predict(X_test)
            cv.append(plsda.score(X_test, y_test))
            cvr2.append(r2_score(plsda.predict(X_train), y_train))

        # Storing results for each number of components
        CVs.append(np.mean(cv))
        CVr2s.append(np.mean(cvr2))

    return {'CVscores':CVs, 'CVR2scores':CVr2s}


def PLSDA_model_CV(df, labels, regres=False, n_comp=10,
                   kf = None, n_fold=5,
                   iter_num=1,
                   encode2as1vector=True,
                   scale=False,
                   feat_type='VIP',
                   random_state=None):
    
    """Perform PLS-DA with n-fold cross-validation.

       df: pandas DataFrame; includes X equivalent in PLS-DA (training vectors).
       labels: target labels.
       regres: bool (default: False); True when the biological problem is a regression and False when it is a classification.
       n_comp: integer; number of components to use in PLS-DA.
       kf: default None; pass a specific cross validation method from 
        https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators (3.1.2).
       n_fold: int (default: 5); number of groups to divide dataset in for cross-validation
        (NOTE: max n_fold can not exceed minimum number of samples per class).
       iter_num: int (default: 1); number of iterations that cross validation is repeated.
       encode2as1vector: bool (default: True); if you have 2 classes, if True, encode them as 0 and 1 in one vector; else
        use one-hot encoding as with multi-class cases.
       scale: bool (default: False); if data is scaled when inputted to PLS model (only true if scaling was not done earlier).
       feat_type: string (default: 'VIP'); types of feature importance metrics to use; accepted: {'VIP', 'Coef', 'Weights'}.
       random_state: int (default: None); set random state for stratified cross_validation (random state is random_state*(current_iteration)).

    Returns: (accuracy, F1-score, precision, recall, Q2, import_features);
        accuracy: list of accuracy values in group selection
        F1-score: list of F1-scores (weighted) in group selection
        precision: list of precision (weighted) in group selection
        recall: list of recall (weighted) in group selection
        Q2: list of average Q2 scores of the models
        imp_features: list of tuples (index number of feature, feature importance)
            ordered by decreasing feature importance.
    """

    # Setting up lists and matrices to store results
    CVR2 = []
    if regres:
        msquared_errors = []
    else:
        accuracies = []
        f1_scores = []
        precision = []
        recall = []

    Imp_Feat = np.zeros((iter_num * n_fold, df.shape[1]))
    f = 0

    unique_labels = list(pd.unique(np.array(labels)))

    is1vector = (len(unique_labels) == 2 and encode2as1vector) or regres

    if regres:
        matrix = np.array(labels)
    else:
        matrix = _generate_y_PLSDA(labels, unique_labels, is1vector)

    if is1vector:
        # keep a copy to use later
        target1D = matrix.copy()

    base_kf = kf

    # Number of iterations equal to iter_num
    for i in range(iter_num):
        kf = base_kf
        if kf is None:
            if random_state != None:
                kf = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True, random_state=random_state*(i+1))
            else:
                kf = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True)

        # Setting up storing variables for cross-validation
        nright = 0 # For accuracy
        cvr2 = [] # For R2 score
        # To store real and predicted classes to calculate F1-score, precision and recall
        if not is1vector:
            all_preds = pd.DataFrame(columns=matrix.columns, index=matrix.index)
            all_tests = pd.DataFrame(columns=matrix.columns, index=matrix.index)
            a = 0
        else:
            all_preds = []
            all_tests = []

        # Iterate through cross-validation procedure
        for train_index, test_index in kf.split(df, labels):
            plsda = PLSRegression(n_components=n_comp, scale=scale)
            X_train, X_test = df.iloc[train_index, :], df.iloc[test_index, :]
            if not is1vector:
                y_train = matrix.iloc[train_index, :].copy()
                y_test = matrix.iloc[test_index, :].copy()

            else:
                y_train, y_test = target1D[train_index], target1D[test_index]
                correct = target1D[test_index]

            # Fit PLS model
            plsda.fit(X=X_train, Y=y_train)

            # Obtain results with the test group
            y_pred = plsda.predict(X_test)
            cvr2.append(r2_score(y_test, y_pred))

            if regres:
                # Save y-test and predictions to calculate F1-score, precision and recall
                all_preds.extend(list(y_pred))
                all_tests.extend(y_test)
            else:
                # Decision rule for classification
                # Decision rule chosen: sample belongs to group where it has max y_pred (closer to 1)
                # In case of 1,0 encoding for two groups, round to nearest integer to compare
                if not is1vector:
                    rounded_pred = y_pred.copy()
                    for i in range(len(y_pred)):
                        if list(y_test.iloc[i, :]).index(max(y_test.iloc[i, :])) == np.argmax(
                            y_pred[i]
                        ):
                            nright += 1  # Correct prediction

                        for l in range(len(y_pred[i])):
                            if l == np.argmax(y_pred[i]):
                                rounded_pred[i, l] = 1
                            else:
                                rounded_pred[i, l] = 0

                    # Save y-test and predictions to calculate F1-score, precision and recall
                    all_tests.iloc[a:a+len(y_test)] = y_test
                    all_preds.iloc[a:a+len(y_test)] = rounded_pred
                    a = a + len(y_test)

                else:
                    rounded = np.round(y_pred)
                    for p in range(len(y_pred)):
                        if rounded[p] >= 1:
                            rounded[p] = 1
                        else:
                            rounded[p] = 0
                        if rounded[p] == correct[p]:
                            nright += 1  # Correct prediction

                    # Save y-test and predictions to calculate F1-score, precision and recall
                    all_preds.extend(list(rounded))
                    all_tests.extend(y_test)

            # Calculate important features (3 different methods to choose from)
            if feat_type == 'VIP':
                Imp_Feat[f, :] = _calculate_vips(plsda)
            elif feat_type == 'Coef':
                Imp_Feat[f, :] = abs(plsda.coef_).sum()
            elif feat_type == 'Weights':
                Imp_Feat[f, :] = abs(plsda.x_weights_).sum(axis=1)
            else:
                raise ValueError(
                    'Type not Recognized. Types accepted: "VIP", "Coef", "Weights"'
                )

            f += 1

        if regres:
            # Calculating the mean squere error of the regression and storing score results
            CVR2.append(np.mean(cvr2))
            # Calculating F1-score, precision and recall for the fold and storing results
            msquared_errors.append(mean_squared_error(all_tests, all_preds))
        else:
            # Calculating the accuracy of the group predicted and storing score results
            accuracies.append(nright / len(labels))
            CVR2.append(np.mean(cvr2))
            # Calculating F1-score, precision and recall for the fold and storing results
            if not is1vector:
                f1_scores.append(f1_score(all_tests.astype(int), all_preds.astype(int), average='weighted'))
                precision.append(precision_score(all_tests.astype(int), all_preds.astype(int), average='weighted'))
                recall.append(recall_score(all_tests.astype(int), all_preds.astype(int), average='weighted'))
            else:
                f1_scores.append(f1_score(all_tests, all_preds, average='weighted'))
                precision.append(precision_score(all_tests, all_preds, average='weighted'))
                recall.append(recall_score(all_tests, all_preds, average='weighted'))


    # Join and sort all important features values from each cross validation group and iteration.
    Imp_sum = Imp_Feat.sum(axis=0) / (iter_num * n_fold)
    imp_features = sorted(enumerate(Imp_sum), key=lambda x: x[1], reverse=True)
    if iter_num == 1:
        if regres:
            return {'mean_squared_error': msquared_errors[0],
                    'Q2': CVR2[0], 'imp_feat': imp_features}
        else:
            return {'accuracy': accuracies[0], 'F1-scores':f1_scores[0], 'precision': precision[0], 'recall':recall[0],
                    'Q2': CVR2[0], 'imp_feat': imp_features}
    else:
        if regres:
            return {'mean_squared_error': msquared_errors,
                'Q2': CVR2, 'imp_feat': imp_features}
        else:
            return {'accuracy': accuracies, 'F1-scores':f1_scores, 'precision': precision, 'recall':recall,
                    'Q2': CVR2, 'imp_feat': imp_features}


def PLSDA_ROC_cv(treated_data, target, pos_label, n_comp=10, scale=False, n_iter=1, cv=None, n_fold=5, random_state=None):
    """Fits and extracts PLS-DA model data and calculates metrics to plot a ROC curve."""

    # Run classifier with cross-validation and plot ROC curves
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    base_cv = cv

    encoded_target = []
    for i in target:
        if i == pos_label:
            encoded_target.append(1)
        else:
            encoded_target.append(0)

    # Number of times PLS-DA cross-validation is made
    # with `n_fold` randomly generated folds.
    for itr in range(n_iter):
        # Evaluate RF model through cross-validation section
        cv = base_cv
        if cv is None:
            if random_state != None:
                cv = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True, random_state=random_state*(itr+1))
            else:
                cv = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True)

        # Fit and evaluate a PLS-DA model for each fold in cross validation
        for train_index, test_index in cv.split(treated_data, encoded_target):
            # Random Forest setup and fit
            plsda = PLSRegression(n_components=n_comp, scale=scale)
            X_train, X_test = treated_data.iloc[train_index, :], treated_data.iloc[test_index, :]
            y_train, y_test = [encoded_target[i] for i in train_index], [encoded_target[i] for i in test_index]
            # Classifier fit
            plsda.fit(X_train, y_train)

            # Metrics for ROC curve plotting
            scores = plsda.predict(X_test)#[:,rf.classes_ == pos_label]

            fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            #interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_score(y_test, scores))

    # Mean of every fold of the cross-validation
    mean_tpr = np.mean(tprs, axis=0)
    #mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # fpr - false positive rate, tpr - true positive rate, AUC - area under curve
    return {'average fpr': mean_fpr, 'average tpr': mean_tpr,
            'upper tpr': tprs_upper, 'lower trp': tprs_lower,
            'mean AUC': mean_auc, 'std AUC': std_auc}


def permutation_PLSDA(df, labels, n_comp=10, iter_num=100, cv=None, n_fold=5, random_state=None,
                      encode2as1vector=True, scale=False, metric='accuracy'):
    """Performs permutation test n times of a dataset for PLS-DA classifiers giving its performance (estimated by
        cross-validation) for the original and all permutations made and respective p-value.

       df: Pandas DataFrame.
       labels: target labels.
       n_comp: int (default - 10); number of components to use in PLS-DA.
       iter_num: int (default - 100); number of permutations made.
       cv: splitter class of sklearn.model_selection (default - None); choose a cross-validation method (and respective
        args); if None, the default method is stratified cross-validation.
       n_fold: int (default - 3); number of groups to divide dataset in for k-fold cross-validation (max n_fold =
        minimum number of samples belonging to one group) if cv is None.
       random_state: int (default - None); random seed given to make the permutations rng class labels.
       encode2as1vector: bool (default: True); if you have 2 classes, if True, encode them as 0 and 1 in one vector; else
        use one-hot encoding as with multi-class cases.
       scale: bool (default: False); if data is scaled when inputted to PLS model (only true if scaling was not done earlier).
       metric: str (default - 'accuracy'); metric to give to scikit-learn cross_validate. Accepted: 'accuracy',
        'f1_weighted', 'recall_weighted' or 'precision_weighted'.

       Returns: (scalar, list of scalars, scalar);
        estimated predictive accuracy of the non-permuted Random Forest model
        estimated predictive accuracy of all permuted Random Forest models
        p-value ((number of permutations with accuracy > original accuracy) + 1)/(number of permutations + 1).
    """

    # get a bit generator
    rng = np.random.default_rng(seed=random_state)

    # list to store results
    Accuracy = []
    f1_scores = []
    precision = []
    recall = []
    if metric not in ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']:
        raise ValueError("Metric not accepted. Must be one of 'accuracy', 'f1_weighted', 'recall_weighted' or 'precision_weighted'.")

    # list of rows to shuffle and dataframe of the data to put rows in each NewC shuffled order
    NewC = np.arange(df.shape[0])
    df = df.copy()  # TODO: check if this copy is really necessary

    unique_labels = list(pd.unique(np.array(labels)))

    is1vector = len(unique_labels) == 2 and encode2as1vector

    matrix = _generate_y_PLSDA(labels, unique_labels, is1vector)

    if is1vector:
        # keep a copy to use later
        correct_labels = matrix.copy()

    # For dividing the dataset in balanced n_fold groups with a random random state maintained
    # in all permutations (identical splits)
    if cv is None:
        cv = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True)

    # Number of permutations + dataset with non-shuffled labels equal to iter_num + 1
    for i in range(iter_num + 1):
        # Temporary dataframe with rows in order of the NewC
        temp = df.iloc[NewC, :]

        # Setting up variables for results of the application of n-fold cross-validated PLS-DA
        nright = 0

        # To store real and predicted classes to calculate F1-score, precision and recall
        if not is1vector:
            all_preds = pd.DataFrame(columns=matrix.columns, index=matrix.index)
            all_tests = pd.DataFrame(columns=matrix.columns, index=matrix.index)
            a = 0
        else:
            all_preds = []
            all_tests = []

        # Repeating for each of the n groups
        for train_index, test_index in cv.split(df, labels):
            # plsda model building for each of the n stratified groups made
            plsda = PLSRegression(n_components=n_comp, scale=scale)
            X_train, X_test = temp.iloc[train_index, :], temp.iloc[test_index, :]
            if not is1vector:
                y_train = matrix.iloc[train_index, :].copy()
                y_test = matrix.iloc[test_index, :].copy()

            else:
                y_train, y_test = correct_labels[train_index], correct_labels[test_index]
                correct = correct_labels[test_index]

            # Fitting the model
            plsda.fit(X=X_train, Y=y_train)

            # Predictions the test group
            y_pred = plsda.predict(X_test)

            # Decision rule for classification
            if not is1vector:
                rounded_pred = y_pred.copy()
                for i in range(len(y_pred)):
                    if list(y_test.iloc[i, :]).index(max(y_test.iloc[i, :])) == np.argmax(
                        y_pred[i]
                    ):
                        nright += 1  # Correct prediction

                    for l in range(len(y_pred[i])):
                        if l == np.argmax(y_pred[i]):
                            rounded_pred[i, l] = 1
                        else:
                            rounded_pred[i, l] = 0

                # Save y-test and predictions to calculate F1-score, precision and recall
                all_tests.iloc[a:a+len(y_test)] = y_test
                all_preds.iloc[a:a+len(y_test)] = rounded_pred
                a = a + len(y_test)

            else:
                rounded = np.round(y_pred)
                for p in range(len(y_pred)):
                    if rounded[p] >= 1:
                        rounded[p] = 1
                    else:
                        rounded[p] = 0
                    if rounded[p] == correct[p]:
                        nright += 1  # Correct prediction

                # Save y-test and predictions to calculate F1-score, precision and recall
                all_preds.extend(list(rounded[:,0]))
                all_tests.extend(y_test)


        # Calculate accuracy for this iteration
        Accuracy.append(nright / len(labels))
        # Calculate F1-score, precision and recall for the fold and storing results
        if metric in ['f1_weighted', 'recall_weighted', 'precision_weighted']:
            if not is1vector:
                f1_scores.append(f1_score(all_tests.astype(int), all_preds.astype(int), average='weighted'))
                precision.append(precision_score(all_tests.astype(int), all_preds.astype(int), average='weighted'))
                recall.append(recall_score(all_tests.astype(int), all_preds.astype(int), average='weighted'))
            else:
                f1_scores.append(f1_score(all_tests, all_preds, average='weighted'))
                precision.append(precision_score(all_tests, all_preds, average='weighted'))
                recall.append(recall_score(all_tests, all_preds, average='weighted'))

        # Shuffle dataset rows, generating 1 permutation of the labels
        rng.shuffle(NewC)

    if metric == 'accuracy':
        performance = Accuracy.copy()
    elif metric == 'f1_weighted':
        performance = f1_scores.copy()
    elif metric == 'recall_weighted':
        performance = recall.copy()
    elif metric == 'precision_weighted':
        performance = precision.copy()

    # Return also the K-fold cross-validation performance for the non-shuffled dataset
    # and the p-value
    CV = performance[0] # Predictive Accuracy of non-permuted dataset PLS-DA model - performance[0]
    pvalue = (
        sum( [performance[i] for i in range(1, len(performance)) if performance[i] >= performance[0]] ) + 1
    ) / (iter_num + 1)

    return CV, performance[1:], pvalue


### Step 6 Functions
### Functions related to perform Univariate Analysis

### These functions are altered versions from the ones available in the multianalysis.py file

# Compute Fold Changes
def computeFC(data, labels, control_class, test_class):
    "Computes Fold-change of features between test and control class."
    # NOTE: labels must be given explicitly, now

    unique_labels = pd.unique(np.array(labels))
    if len(unique_labels) != 2:
        raise ValueError('The number of groups in the data is not two')

    locs0 = [i for i, lbl in enumerate(labels) if lbl == test_class]
    locs1 = [i for i, lbl in enumerate(labels) if lbl == control_class]

    g0means = data.iloc[locs0, :].mean(axis=0)#.replace({np.nan:0})
    g1means = data.iloc[locs1, :].mean(axis=0)#.replace({np.nan:0})

    FC = g0means / g1means
    FC.name = f'FC ({test_class} / {control_class})'

    log2FC = np.log2(FC)
    log2FC.name = 'log2FC'

    return pd.concat([FC, log2FC], axis=1)


# Compute p-values between two classes
def compute_pvalues_2groups(data, labels, control_class, test_class, equal_var=True, useMW=False):
    "Computes p-values based on t-test or Mann-Whitney tests between test and control class."
    unique_labels = pd.unique(np.array(labels))
    if len(unique_labels) != 2:
        raise ValueError('The number of groups in the data is not two')
    locs0 = [i for i, lbl in enumerate(labels) if lbl == test_class]
    locs1 = [i for i, lbl in enumerate(labels) if lbl == control_class]

    pvalues = []
    for i, col in enumerate(data.columns):
        v0, v1 = data.iloc[locs0, i], data.iloc[locs1, i]
        if not useMW:
            tx = stats.ttest_ind(v0, v1, equal_var=equal_var)
        else:
            tx = stats.mannwhitneyu(v0, v1, alternative='two-sided')
        pvalues.append(tx.pvalue)

    pvalues = pd.Series(pvalues, index=data.columns, name='p-value').sort_values()
    adjusted = p_adjust_bh(pvalues.values) # TODO: optionally use other methods
    adjusted = pd.Series(adjusted, index=pvalues.index, name='FDR adjusted p-value')
    return pd.concat([pvalues, adjusted], axis=1)

def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing.

       From answer in StOvf
       https://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python"""

    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def compute_FC_pvalues_2groups(normalized, processed,
                               labels, control_class, test_class,
                               equal_var=True, useMW=False):
    "Computes feature Fold-change and p-values (univariate tests) between test and control class."
    FC = computeFC(normalized, labels, control_class, test_class)
    pvalues = compute_pvalues_2groups(processed, labels,
                                      control_class, test_class,
                                      equal_var=equal_var, useMW=useMW)
    FC = FC.reindex(pvalues.index)
    return pd.concat([pvalues, FC], axis=1)

def compute_pvalues_multiple_groups(data, groups, useKW=False):
    "Computes ANOVA and Kruskal-Wallis test p-values."
    
    pvalues = []

    for col in data.columns:
        col_data = data[col]
        samples = []
        for g in groups:
            samples.append(col_data.loc[groups[g]])
        if not useKW:
            tx = stats.f_oneway(*samples)
        else:
            tx = stats.kruskal(*samples)
        pvalues.append(tx.pvalue)

    pvalues = pd.Series(pvalues, index=data.columns, name='p-value').sort_values()
    adjusted = p_adjust_bh(pvalues.values) # TODO: optionally use other methods
    adjusted = pd.Series(adjusted, index=pvalues.index, name='FDR adjusted p-value')
    return pd.concat([pvalues, adjusted], axis=1)


### Step 8 Functions
### Functions for Van Krevelen Diagrams, Kendrick Mass Defect Plots and Chemical Composition Series

elem_pattern = r'[A-Z][a-z]?\d*'
elem_groups = r'([A-Z][a-z]?)(\d*)'

def element_composition(formula, elements=None):
    """Given a string with a formula, return dictionary of element composition."""

    composition = {}
    for elemp in re.findall(elem_pattern, formula):
        match = re.match(elem_groups, elemp)
        n = match.group(2)
        number = int(n) if n != '' else 1
        composition[match.group(1)] = number

    if elements is None:
        return composition

    return {e : composition.get(e, 0) for e in elements}


default_series = ('CHO', 'CHOS', 'CHON', 'CHNS', 'CHONS', 'CHOP', 'CHONP','CHONSP')

def create_element_counts(data, formula_subset=['Formula',],
                          compute_ratios=True,
                          series=default_series,
                          drop_duplicates=True):
    """Create DataFrame from element counts and concat to original DataFrame.

       Optionally, the ratios of H/C and O/C and element composition series are also computed"""

    print("------ create elem counts")
    print(data.head(5))

    # safe guard: remove empty formulae
    if isinstance(formula_subset, str):
        formula_subset = [formula_subset]
    
    formulae = data[formula_subset]

    # count elements
    forms_list = []
    idxs_list = []
    for col in formulae.columns:
        formulae_col = formulae[[col]].dropna()
        for indx in formulae_col.index:
            fs = formulae_col.loc[indx].iloc[0]
            if isinstance(fs, str): # For Smart Formula and str based annotations
                forms_list.append(fs)
                idxs_list.append(indx)

            else: # For meta_cols and list based annotations
                l_unique = set(fs)
                for f in l_unique:
                    #print(f)
                    forms_list.append(f)
                    idxs_list.append(indx)
    ecounts_list = []
    for f in forms_list:
        ecounts_list.append(element_composition(f))

    result = pd.DataFrame(ecounts_list).fillna(0).astype(int)
    result['idxs'] = idxs_list

    # compute ratios for VK plots
    if compute_ratios:
        result['H/C'] = result['H'] / result['C']
        result['O/C'] = result['O'] / result['C']

    # compute series from compositions

    sorted_series = [''.join(sorted(list(s))) for s in series]
    result_series = []

    for composition in ecounts_list:
        nonzero = ''.join(sorted([k for k, c in composition.items() if c > 0]))

        if nonzero in sorted_series:
            result_series.append(series[sorted_series.index(nonzero)])
        else:
            result_series.append('other')

    result['Series'] = pd.Series(result_series, index=result.index)
    if not isinstance(formula_subset, str):
        result = result.set_index('idxs')
        if drop_duplicates:
            result = result.drop_duplicates()
    
    return result

def calc_kmd(data, rounding='up', neutral_mass_col='Neutral Mass'):
    """Calculates Kendrick Mass Nominal Mass and Kendrick Mass Defect to plot Kendrick Mass Defect Plots.

       rounding: str (default: 'up') - 'up' or 'nearest', determines what nominal mass to consider, if rounding Kendrick mass
        up or to the nearest integer.

       returns: the Nominal Kendrick Mass and the Kendrick Mass Defect for the mass data given."""

    # Calculating Exact Kendrick Mass
    masses = data[neutral_mass_col]
    Kendrick_m = masses * 14 / 14.0156500638

    nominal = []
    fraction = []

    # Calculating Nominal Kendrick Mass and Kendrick Mass Defect
    for i in Kendrick_m:
        if rounding  == 'nearest':
            s = np.round(i) # Rounding 
            nominal.append(s) # Nominal Kendrick Mass
            fraction.append(s-i)

        elif rounding == 'up':
            frac, whole = np.modf(i)
            s = whole + 1 # Rounding up
            nominal.append(s) # Nominal Kendrick Mass
            fraction.append(s-i) # Kendrick Mass Defect

        else:
            raise ValueError("Rounding method not accepted. Available methods are: 'nearest' or 'up'.")

    return nominal, fraction
