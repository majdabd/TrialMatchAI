import joblib
from tqdm.auto import tqdm
from preprocessing_utils import eic_text_preprocessing
import pandas as pd
import os

memory = joblib.Memory(".")
def ParallelExecutor(use_bar="tqdm", **joblib_args):
    """Utility for tqdm progress bar in joblib.Parallel"""
    all_bar_funcs = {
        "tqdm": lambda args: lambda x: tqdm(x, **args),
        "False": lambda args: iter,
        "None": lambda args: iter,
    }
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type" % bar)
            # Pass n_jobs from joblib_args
            return joblib.Parallel(n_jobs=joblib_args.get("n_jobs", 10))(bar_func(op_iter))

        return tmp
    return aprun


class Preprocessor:
    def __init__(self, id_list, n_jobs):
        self.id_list = id_list
        self.n_jobs = n_jobs
    

    def preprocess_clinical_trials_text(self):
        parallel_runner = ParallelExecutor(n_jobs=self.n_jobs)(total=len(self.id_list))
        X = parallel_runner(
            joblib.delayed(eic_text_preprocessing)(
            [nct_id]
            )
            for nct_id in self.id_list
        )     
        return pd.concat(X).reset_index(drop=True)
        