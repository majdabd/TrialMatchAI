import requests
import xml.etree.ElementTree as ET
import os
import time
import json
import re
import gzip, tarfile
import joblib 
from tqdm.auto import tqdm
import numpy as np

def normalize_whitespace(s):
    return ' '.join(s.split())

def download_study_info(nct_id, runs=2):
    local_file_path = f"../data/trials_xmls/{nct_id}.xml"
    updated_cts = []
    for run in range(runs):
        if os.path.exists(local_file_path):
            # Read the content of the existing local XML file
            with open(local_file_path, "r") as f:
                local_xml_content = f.read()
            try:
                local_root = ET.fromstring(local_xml_content)
            except ET.ParseError as e:
                print(f"Error parsing XML for trial {nct_id}: {e}")
                os.remove(local_file_path)
                continue
            
            # Download the online version of the XML
            url = f"https://clinicaltrials.gov/ct2/show/{nct_id}?displayxml=true"
            response = requests.get(url)
            
            if response.status_code == 200:
                online_xml_content = response.text
                # Parse the XML content
                online_root = ET.fromstring(online_xml_content)
                to_check = ["eligibility", "brief_title", "overall_status", "location"]
                
                local_version = []
                online_version = []
                
                for s in to_check:
                    local_elem = local_root.find(".//%s" % s)
                    online_elem = online_root.find(".//%s" % s)
                    
                    # Check if the element exists in both versions
                    if local_elem is not None and online_elem is not None:
                        local_version.append(local_elem)
                        online_version.append(online_elem)
                    else:
                        continue
                
                is_updated = any([normalize_whitespace(ET.tostring(a, encoding='unicode').strip()) !=
                                normalize_whitespace(ET.tostring(b, encoding='unicode').strip())
                                for a, b in zip(local_version, online_version)])
                
                if is_updated:
                    updated_cts.append(nct_id)
                    # Update the local XML with the online version
                    with open(local_file_path, "w") as f:
                        f.write(ET.tostring(online_root, encoding='unicode'))
                    print(f"Updated eligibility criteria for {nct_id}")
                else:
                    if run == 1:
                        print(f"No changes in eligibility criteria for {nct_id}.")
            else:
                print(f"Error downloading study information for {nct_id}")
        else:
            downloaded = False
            while not downloaded:
                url = f"https://clinicaltrials.gov/ct2/show/{nct_id}?displayxml=true"
                response = requests.get(url)
                if response.status_code == 200:
                    root = ET.fromstring(response.text)
                    with open(local_file_path, "w") as f:
                        f.write(ET.tostring(root, encoding='unicode'))
                    downloaded = True
                    print(f"Study information downloaded for {nct_id}")
                else:
                    print(f"Error downloading study information for {nct_id}")
                
                if not downloaded:
                    print(f'Download of {nct_id}.xml failed. Retrying in 2 seconds...')
                    time.sleep(2)
    return updated_cts

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

def parallel_downloader(
    n_jobs,
    nct_ids,
):
    parallel_runner = ParallelExecutor(n_jobs=n_jobs)(total=len(nct_ids))
    X = parallel_runner(
        joblib.delayed(download_study_info)(
        nct_id, 
        )
        for nct_id in nct_ids
    )     
    updated_cts = np.vstack(X).flatten()
    return updated_cts 


class Downloader:
    def __init__(self, id_list, n_jobs):
        self.id_list = id_list
        self.n_jobs = n_jobs
    def download_and_update_trials(self):
        print(f"Downloading and updating information for {len(self.id_list)} trials...")
        start_time = time.time()

        updated_cts = parallel_downloader(self.n_jobs, self.id_list)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Download and/or update completed in {elapsed_time:.2f} seconds.")
    