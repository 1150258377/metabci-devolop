# -*- coding: utf-8 -*-
"""
Emotion recognition paradigm.
"""
from typing import Optional, Union, List, Dict, Tuple
import numpy as np
import mne
from .base import BaseParadigm

class EmotionParadigm(BaseParadigm):
    """Emotion recognition paradigm.
    
    This paradigm is designed for emotion recognition tasks using EEG data.
    It supports both classification (discrete emotions) and regression (continuous
    emotion dimensions) tasks.
    """
    
    def __init__(
        self,
        channels: Optional[List[str]] = None,
        events: Optional[Dict[str, Tuple[int, Tuple[float, float]]]] = None,
        srate: Optional[float] = None,
        interval: Optional[List[float]] = None,
        **kwargs
    ):
        super().__init__(
            channels=channels,
            events=events,
            srate=srate,
            interval=interval,
            **kwargs
        )
        
    def is_valid(self, dataset):
        """Check if the dataset is valid for this paradigm."""
        return True  # For now, accept all datasets
        
    def _get_single_subject_data(self, dataset, subject_id, verbose=False):
        """Get data for a single subject."""
        return super()._get_single_subject_data(dataset, subject_id, verbose)
        
    def get_data(
        self,
        dataset,
        subjects: Optional[List[Union[str, int]]] = None,
        return_concat: bool = False,
        n_jobs: Optional[int] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ):
        """Get data from the dataset.
        
        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance.
        subjects : List[Union[str, int]], optional
            List of subjects to get data for. If None, use all subjects.
        return_concat : bool, optional
            If True, concatenate all data into a single array. Default is False.
        n_jobs : int, optional
            Number of jobs to run in parallel. If None, use all available cores.
        verbose : bool | str | int, optional
            Level of verbosity.
            
        Returns
        -------
        X : ndarray
            EEG data.
        y : ndarray
            Labels.
        meta : DataFrame
            Metadata.
        """
        return super().get_data(
            dataset,
            subjects=subjects,
            return_concat=return_concat,
            n_jobs=n_jobs,
            verbose=verbose
        ) 