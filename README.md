# Revealing Important Metabolic Subnetworks Through Impact in Class Probabilities of Graph Neural Network Classifiers

Reproducible (as possible) code for the paper 'Revealing Important Metabolic Subnetworks Through Impact in Class Probabilities of Graph Neural Network Classifiers' for the development and validation of the Formula-Difference Graph Neural Networks (FDiGNN) approach.

Here we present the code used to obtain the different results (including supplementary material and a few extra) presented in the aformentioned paper. Here, we explain the organization of the git-hub repository, what is present in each of the files shown here, the already stored results and what we recommend to do to either re-run the analysis or obtain as similar results as possible to the ones presented.

### Neccessary Python Requirements

The file `requirements.txt` contains all packages used and can be used to create a python environment to run the Python code in this package (instructions in the first 3 lines of the file).


#### Benchmark Datasets Required

The FDiGNN approach was tested on 3 benchmark metabolomics datasets. As such, the analysis is repeated 3 times with the filenames between them being equal except for the sinalization of the benchmark dataset being analysed. The datasets are:

- **LGD** - Liver Graft Dataset (FT-ICR-MS), data in `LGD_Dataset.XLSX` (from https://pubmed.ncbi.nlm.nih.gov/27835640/). This file is in the repository.
- **LID** - Lung Implant Dataset (FT-ICR-MS), data in `ST003338_AN005470_Results.txt` and `ST003338_AN005470.txt` from the Metabolomics Workbench, project ID PR002075 (study ID: ST003338; analysis ID: AN005470). **The file `ST003338_AN005470_Results.txt` can be obtained in https://www.metabolomicsworkbench.org/data/DRCCStudySummary.php?Mode=SetupRawDataDownload&StudyID=ST003338 by clicking the button with the same name (`ST003338_AN005470_Results.txt`). The file `ST003338_AN005470.txt` can be obtained in https://www.metabolomicsworkbench.org/data/study_textformat_list.php?STUDY_ID=ST003338&STUDY_TYPE=MS&RESULT_TYPE=5 by clicking the `Download mwTab` button on the Positive Ion Mode analysis. The 2 files must be placed in the same folder as this repository.**
- **MD** - Malaria Dataset (LC-MS), data in `ST000578_AN000888_Results.txt` from the Metabolomics Workbench, project ID PR000384 (study ID: ST000578; analysis ID: AN000888). **This file can be obtained in https://www.metabolomicsworkbench.org/data/DRCCStudySummary.php?Mode=SetupRawDataDownload&StudyID=ST000578 by clicking the button `ST000578_AN000888_Results.txt`. Place the downloaded file in the same folder as this repository.**

A description of each dataset is presented at the beginning of relevant jupyter notebooks that will use said dataset. LID and MD are publicly available at the NIH Common Fund’s National Metabolomics Data Repository website, the Metabolomics Workbench, for LID with project ID PR002075 (study ID: ST003338; analysis ID: AN005470) and for MD with project ID PR000384 (study ID: ST000578; analysis ID: AN000888) supported by the Metabolomics Workbench/National Metabolomics Data Repository (NMDR) (grant# U2C-DK119886), Common Fund Data Ecosystem (CFDE) (grant# 3OT2OD030544) and Metabolomics Consortium Coordinating Center (M3C) (grant# 1U2C-DK119889).


#### Database Files

- HMDB Database (for annotation) - `hmdb_complete.xlsx`
- SMPDB Database (Small Molecule Pathway Database) used for metabolic pathway information - `smpdb_pathways.csv`


## `Dataset_FDiGNN_DataTreatment.ipynb` Files

`LGD_FDiGNN_DataTreatment.ipynb`, `LID_FDiGNN_DataTreatment.ipynb`, `MD_FDiGNN_DataTreatment.ipynb` Files.

In these files, the original data is read and data organization, feature filtering, pre-processing, pre-treatment, data annotation and formula assignments are performed. Furthermore, a quick analysis on dataset characteristics (Sup. Table 1) and unsupervised analysis is performed (Sup. Figure 1). **LID and MD datasets have to be downloaded from the Metabolomics Workbench for their notebooks to be runable. See `Benchmark Datasets Required` section above.**

The treated datasets are then outputted in 4 different files (for each dataset) that are used as inputs for the remaining analysis files. These files are stored in the `Data` folder as well as in the `Data` folder for each datasets' corresponding `Simulations` folder. As such, it is **not necessary to run this file to run the remaining files**. The generated files are:

- `Dataset_TreatedData_Final.pickle` containing the data resulting from the fully treated data that will be used extensively in other analysis.
- `Dataset_Target_Final.pickle` containing the target (the class labels of each sample) of the dataset.
- `Dataset_ProcData_Final.pickle` containing the data after normalization without missing value imputation performed beforehand or other data pre-treatments used after (transformations, scaling) - used for the intensity bar plots and boxplots.
- `Dataset_AllTreatedData_Final.xlsx` containing a sheet with metadata and normalized (non-imputed) data, another with fully treated data, another with binary simplified data and another with only missing value imputation and normalized data (for the `MD_AllTreatedData_Final.xlsx` to be smaller than 100 Mb and be added to this repository, this latter sheet was manually removed since it is not required in further analyses).

To perform the formula assignment, the algorithm used requires a formula database in the folder to be ran. As this database was too large to include in the git-hub repository, **the file `FormulaDatabaseCreator.ipynb` (creates the formula database from scratch) needs to be ran before running the DataTreatment files**. It is a jupyter notebook that should already have all the conditions to mimic the formula database used.

The `Paper_Figs` folder contains the PCA plots used in Sup. Figure 1 for each dataset (`LGD_PCAplot.png`, `LID_PCAplot.png`, `MD_PCAplot.png`) so the file does not need to be ran.


## `Dataset_FDiGNN_Performance_FeatImp` Files

`LGD_FDiGNN_Performance_FeatImp`, `LID_FDiGNN_Performance_FeatImp`, `MD_FDiGNN_Performance_FeatImp` Files.

In these files, Formula-Difference networks (FDiNs) are built, FDiGNN models are set up and fitted for both comparing model performance to RF and PLS-DA model as well as comparing important metabolites to build these models. Furthermore, network and pathway enrichment analysis for each dataset is performed ending with an interactive dash app to observe your data in the network context. As such, this is the basis for Figures 4 and 8, Tables 1 to 4 and Sup. Tables 2 and 3 in the paper.

At the end of creating the FDiNs, the FDiN of each dataset is saved as `LGD_FDiN_Final.pickle`, `LID_FDiN_Final.pickle`, `MD_FDiN_Final.pickle`.

For estimating model performance, no models are saved for LGD and LID, thus slightly different results can be obtained. As estimating model performance with 10 iterations takes a while, the `see_model_performance` parameter is initially set to False. Change to True to run. For MD, the 10 iterations of models are already run using the train/test splits with the same exact random_seed as set in the file and stored in the `Models` folder (`MD_model_4TAG_64HC3D005LR0001WD3K_{0-9}`). If the parameter `fit_models` is False, these models will be loaded, else others will be fitted and **will replace in the `Models` folder** the pre-trained weights.

To **obtain the same FDiGNN results as in the paper for node importance**, pretrained model weights are saved for each dataset in the `Models` folder to load (LGD - `LGD_model_4TAG_64HC3D005LR0001WD140E`, LID - `LID_model_4TAG_64HC15D001LR0001WD`, MD - `MD_model_4TAG_64HC3D005LR0001WD3K_0`). If `model_fitting` is set to False, these models are loaded (default), if set to True, **a new model will be trained under the same conditions and will be saved, replacing the aforementioned saved weights**.

At the end of these files, a section to obtain a list of nodes to act as the pathway graphlets nodes, single nodes and possible gap nodes within the pathway graphlets nodes is present. This creates sets of 20 or 16 combinations of these for 4-length pathways + 5 single nodes, 5-length pathways + 5 single nodes or 6-length pathways + 5 single nodes. The sets created are then saved in the `Data` folder of the corresponding `Simulations` folder of the dataset with the names:

- `Dataset_GNNPathwayTest_graphlets_size5.txt` for the 5-length pathway nodes.
- `Dataset_GNNPathwayTest_singlenodes_size5.txt` for the 5 single nodes accompanying the 5-length pathway nodes.
- `Dataset_5Graphlets_gaps.txt` to identify the gaps in the 5-length pathway nodes.


- `Dataset_GNNPathwayTest_graphlets_size6.txt` for the 6-length pathway nodes.
- `Dataset_GNNPathwayTest_singlenodes_size6.txt` for the 5 single nodes accompanying the 6-length pathway nodes.
- `Dataset_6Graphlets_gaps.txt` to identify the gaps in the 6-length pathway nodes.


- `Dataset_GNNPathwayTest_graphlets_size4.txt` for the 4-length pathway nodes.
- `Dataset_GNNPathwayTest_singlenodes_size4.txt` for the 5 single nodes accompanying the 4-length pathway nodes.

The set of nodes used to obtain the results in the paper are already saved. **Every time the file is ran, a different set of graphlets are generated. Therefore, the parameter `obtain_new_graphlets` is set to False as to not overwrite the used graphlets.** These graphlets were used to generate the pre-saved models that will be mentioned in the next section to facilitate obtaining the results shown in the paper. **These model cannot be used if the sets of nodes stored is different as it will lead to WRONG results.**

**Therefore, we STRONGLY recommend to not change the `obtain_new_graphlets` parameter to True.**


## `Simulations` Folders

`LGD_Simulations`, `LID_Simulations`, `MD_Simulations` Folders.

In these folders, the code regarding the simulations performed with modified datasets are shown. Almost all of these were performed on the supercomputer Deucalion cluster under the project 2024.06915.CPCA.A1. Thus, they are presented as they were ran in the Deucalion cluster with a few adjustments in terms of naming so it is better understood what each file is.

They all have 4 folders: `Data` whose contents were previously explained, `Dataset_TAG_Models` (`LGD_Models`for LGD) which contain the fitted models for each simulation so the results are reproducible for the same set of graphlets, `Dataset_TAG_Results` where results will be stored (results are not already here since all the files combined are too large to store in a git-hub repository) and a `ShellScript_Files` folder with the shell script files used to perform the simulation on the Deucalion cluster are stored (with slight modifications removing the name of the partition used and the project ID).

The folder has files relating to the fitting of the models across simulations and another set of files relating to the computing of node/metabolite importances.

#### Fitting Models

These files obtain the model weights that define the FDiGNN models for all simulations. These are already stored in `Dataset_TAG_Models`, therefore **this does not need to be ran (and should not) to obtain the results shown in the paper.** There are 7 combinations of simulations each with either 16 or 20 sets of pathway+single node combinations each represented by a file:

- `Dataset_Graphlets4.py` or `Dataset_Graphlets4_Normal.py` - to run with the 4 length pathway simulations.
- `Dataset_Graphlets5.py` or `Dataset_Graphlets5_Normal.py` - to run with the 5 length pathway simulations.
- `Dataset_Graphlets5_Gap.py` - to run with the 5 length pathway simulations with a neutral gap.
- `Dataset_Graphlets5_OppGap.py` - to run with the 5 length pathway simulations with an opposite gap.
- `Dataset_Graphlets6.py` or `Dataset_Graphlets6_Normal.py` - to run with the 6 length pathway simulations.
- `Dataset_Graphlets6_Gap.py` - to run with the 6 length pathway simulations with a neutral gap.
- `Dataset_Graphlets6_OppGap.py` - to run with the 6 length pathway simulations with an opposite gap.

The `.sh` files of the same names in the `ShellScript_Files` were the ones used in the Deucalion cluster to run these files for all simulations. They can provide guidance on how to write a `.sh` file to run the simulations (the file should be **in the same foler as the `.py` files to run**). To perform a single simulation, a terminal should be open and the following line (similar to the lines in the `.sh` files) should be ran:

**python filename r**, where filename is the name of the file and r is the number of the node set to use (0 to 19 in LGD and 0 to 15 in LID and MD).

Example: **python Dataset_Graphlets5_Gap.py 4**

This will let you obtain the model results for one set of nodes for one of the 7 combinations of simulations which will overwrite the models in the model folder. For running all 7x20 + 7x16x2 = 364 simulations, an automated way is preferred.


#### Node Importance Calculation

These files use the model weights obtained from the prior section to output the prediction impact of each metabolite/node into the `Dataset_TAG_Results`. If ran with the **same saved model weights and pathway+single node sets, it will to same results as in the paper**.

For LGD, the calculation can be performed for all simulations by running the `LGD_NodeImp_Calc.ipynb` file.

For LID and MD, a similar scheme to the model fitting was performed with 7 files covering the 7 combinations of simulations each with either 16 or 20 sets of pathway+single node combinations each represented by a file:

- `Dataset_Graphlets4_NodeImp.py` - to run with the 4 length pathway simulations.
- `Dataset_Graphlets5_NodeImp.py` - to run with the 5 length pathway simulations.
- `Dataset_Graphlets5_Gap_NodeImp.py` - to run with the 5 length pathway simulations with a neutral gap.
- `Dataset_Graphlets5_OppGap_NodeImp.py` - to run with the 5 length pathway simulations with an opposite gap.
- `Dataset_Graphlets6_NodeImp.py` - to run with the 6 length pathway simulations.
- `Dataset_Graphlets6_Gap_NodeImp.py` - to run with the 6 length pathway simulations with a neutral gap.
- `Dataset_Graphlets6_OppGap_NodeImp.py` - to run with the 6 length pathway simulations with an opposite gap.

The `.sh` files of the same names (`Dataset_Graphlets4_NodeImp.sh`, `Dataset_Graphlets5_NodeImp.sh`, `Dataset_Graphlets6_NodeImp.sh`) in the `ShellScript_Files` were the ones used in the Deucalion cluster to run these files for all simulations. They can provide guidance on how to write a `.sh` file to run the simulations (the file should be **in the same foler as the `.py` files to run**). To perform a single simulation, a terminal should be open and the following line (similar to the lines in the `.sh` files) should be ran:

**python filename r**, where filename is the name of the file and r is the number of the node set to use (0 to 15 in LID and MD).

Example: **python Dataset_Graphlets4_NodeImp.py 4**

This should be run for the 16 sets for each of the 7 files and for the 2 datasets to obtain all results (16x7x2=224). For running simulations, an automated way is preferred or an adaptation similar to what is available in `LGD_NodeImp_Calc.ipynb`.


`joining_graphlet_dfs.py` is a file to join part of the results of the different sets of analysis but it is not required.


## `Datasets Results Analysis.ipynb` Files

`LGD Results Analysis.ipynb`, `LID Results Analysis.ipynb`, `MD Results Analysis.ipynb` Files.

In these files, the analysis of the results of all simulations for each dataset is performed leading to Figures 5 to 7 Sup. Figures 2 to 7 and Sup. Table 4. If ran **after all the simulations' node importances have been computed and stored in `Dataset_TAG_Results` for each dataset**, it will provide all tables. **If the simulations models and pathway+single node sets were not overwritted, it will lead to the same figures as presented in the paper.**


## Support files

- `metanalysis_standard.py` and `multianalysis.py` - functions for data filtering, processing, treatment and multivariate statistical analysis.
- `MDiN_functions.py` - functions for building Mass-Difference Networks and to process formulas (strings to dictionary).
- `form_assign_func.py` - functions for both the generation of the formula database as well as the application of the formula assignment algorithm.
- `poly_coefs.json` - coefficients for polynomial functions that are used during formula assignment.
- `elips.py` - functions to draw confidence ellipses in PCA projection plots.


## Other files

- `Metabolic_Network_Build.ipynb` - this notebook was used to generate from the SMPDB database the metabolic knowledge network stored in the file `SMPDB_MetaNetwork_general.pickle`. This notebook **cannot be ran since it does not have the SMPDB database large files used in it in this repository.** It is shown with its outputs to exemplify how the metabolic knowledge network was created.
- `filtered_hmdb.csv` - a filtered HMDB database also used to build the metabolic knowledge network.


# Conclusion

Please, if you have any questions, contact the owners of the repository (aeferreira, fmtraquete).