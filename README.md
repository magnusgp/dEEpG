# Deep Learning in EEG
## 02466 Project Work | s204052, s204075 and s200431
The following is the repository for the Deep Learning in EEG project.
The project is a part of the course 02466 Project work - Bachelor of Artificial Intelligence and Data, which is mandatory for the completion of the B.Sc.Eng programme Artificial Intelligence and Data (Kunstig Intelligens og Data) at the Technical University of Denmark.

The main part of the repository is the pipeline folder. Here, you will find all the utility functions needed for carrying out the experiments described in the 'Experiments' section. The python script pipelineMain.py is the main script for the project and is used to run the whole dynamic pipeline. This is the only file needed to be run in order to carry out the experiments.

All dependencies for the project can be found in and installed with the requirements.txt file. 

To reproduce our results, use the seed 42 and the following dataset. 

## Data
To run the code and the experiments, the TUH EEG Artifact Corpus v2.0.0 is needed. It is freely available at here: 
https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

In order to create the .pkl files for running experiment code, please run the following:
```python
python3 pipelineMain.py --limit 200
```
Where the limit argument is the limit of windows in the downsampled dataset. To run the experiment without limit, use --limit 0

## Preprocessing
To preprocess the raw .edf EEG data files, we use preprocessing tools which are inspired by tools created by David Nyrnberg. Those tools can be found in the following repository:
https://github.com/DavidEnslevNyrnberg/DTU_DL_EEG

## Experiments
The project carried 3 main experiments out:
- **Model Selection Experiment**
```python
python3 clfs.py --classifier "Nearest Neighbors" --n_outer_splits 5 --n_inner_splits 5 --fromPickle 1
```
Runs the Nearst Neighbors classifier with data from a .pkl file (created by the pipelineMain.py call) using GroupKFoldCV with 5 inner and outer splits.
Remember to set the path for the correct .pkl file suitable for the experiment (with 11 files for model selection).

- **Classification Experiment**
```python
python3 clfs.py --classifier "Logistic Regression" --n_outer_splits 5 --n_inner_splits 5 --fromPickle 0
```
Runs the Logistic Regression classifier with data from raw .edf data using GroupKFoldCV with 5 inner and outer splits. 
Remember to set the path for the correct .pkl file suitable for the experiment (with 310 files for model selection).

- **CV Stratification Experiment**


- **Annotation Algorithm Experiment**
```python
python3 visualizations.py
```
Runs the annotation algorithm and visualizes event types for overlap cheecking.

## Credits
The project was carried out by s204052, s204075 and s200431.

We want to thank the following people for their great interest and contributions to this project:

Lars Kai Hansen (lars.kai.hansen@gmail.com)

David Nyrnberg (david@nyrnberg.dk)

Giovanni Grego (s202287@student.dtu.dk)

## License
None
