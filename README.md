# Seizure Prediction from CHB-MIT Scalp EEG

# Project Goal

Build an ML pipeline to predict "siezure vs non-seizure activity" from scalp recordings, using the **CHB-MIT Scalp EEG Database**. 
The longterm goal is to explore seizure **onset prediction** on fixed-length EEG windows and prototype a simple **demo app** using streamlit to visualize predictions. 

## Dataset
**Name:** CHB-MIT Scalp EEG Database
**Source** PhysioNet
**Link:** https://physionet.org/content/chbmit/1.0.0/

Key Characteristics:

–22 pediatrick subjects with intractable seizures (23 cases)
- EEG recorded at 256 Hz, mostly 23 channels (10–20 system)
- Recordings span several days after withdrawal of anti-seizure medication
- 664 EDF files total; 129 contain one or more seizures
- 198 seizures annotated with start and end times

For this project, the initial experiments will use a **subset of 5 patients** and focus on **binary classification (seizure vs non-seizure)** on **fixed-length windows** of EEG (5-second windows).

## High-Level Plan

1. **Day 1: Setup & Dataset Understanding**
   - Define prediction task (seizure vs non-seizure per window)
   - Explore CHB-MIT metadata and EDF structure
   - Design preprocessing and windowing strategy

2. **Day 2: Preprocessing & Feature Extraction**
   - Load EDF files and seizure annotations
   - Filter and normalize EEG signals
   - Segment into fixed-length windows and assign labels

3. **Day 3: Baseline Models**
   - Train classical models (e.g., random forest, simple CNN)
   - Evaluate using subject-wise splits

4. **Day 4-5: Deep Learning + Patient-Specific Models**
   - Implement CNN/LSTM or temporal CNN architectures
   - Compare cross-patient vs patient-specific performance

5. **Day 6: App & Documentation**
   - Build a small Streamlit app to visualize EEG and predictions
   - Document methodology, results, and next steps

## Tech Stack

- **Python**, **PyTorch**, **NumPy**, **pandas**
- **MNE** for EEG file handling and preprocessing
- **scikit-learn** for classical ML and evaluation
- **Streamlit** for interactive demo UI
