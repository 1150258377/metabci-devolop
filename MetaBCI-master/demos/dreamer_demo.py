# -*- coding: utf-8 -*-
"""
Demo script for using the DREAMER dataset with MetaBCI.
"""
import numpy as np
import mne
from metabci.brainda.datasets.dreamer import DREAMER
from metabci.brainda.paradigms.emotion import EmotionParadigm

def main():
    # Initialize the DREAMER dataset
    dataset = DREAMER()
    
    # Get data for subject 1
    subject = 1
    
    # Create an emotion paradigm
    paradigm = EmotionParadigm(
        channels=dataset.channels,
        events=dataset.events,
        srate=dataset.srate,
        interval=[0, 4]  # 4-second epochs
    )
    
    # Get the data
    X, y, meta = paradigm.get_data(
        dataset,
        subjects=[subject],
        return_concat=True,
        n_jobs=1,
        verbose=False
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Meta information:\n{meta.head()}")
    
    # Example of basic preprocessing
    # Filter the data
    X = mne.filter.filter_data(
        X,
        sfreq=dataset.srate,
        l_freq=1,
        h_freq=40,
        verbose=False
    )
    
    print("\nAfter filtering:")
    print(f"Data shape: {X.shape}")
    
    # You can now use this data for your analysis
    # For example, you could:
    # 1. Extract features
    # 2. Train a classifier
    # 3. Perform cross-validation
    # etc.

if __name__ == "__main__":
    main() 