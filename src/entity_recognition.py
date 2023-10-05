import os
os.environ['TRANSFORMERS_CACHE'] = 'models/cache/'

import joblib
from tqdm.auto import tqdm
import requests
from typing import List, Dict, Union
import requests
import numpy as np
import pandas as pd
import glob

import medspacy
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language
from spacy.util import filter_spans
from spacy.tokens import Doc, Token
from spacy.matcher import Matcher
from srsly import read_json
import re
from utils import get_dictionaries_with_values
import transformers
from transformers import AutoTokenizer, pipeline

INPUT_FILEPATH_CT = "../data/preprocessed_data/"
OUTPUT_FILEPATH_CT = "../data/ner_clinical_trials/"
AUXILIARY_ENTITIES_LIST = ["Sign_symptom", "Biological_structure", "Date", "Duration", "Frequency", "Severity", "Lab_value", 
                           "Diagnostic_procedure", "Therapeutic_procedure", "Personal_background", "Clinical_event", "Outcome"]

aux_tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
aux_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", tokenizer= aux_tokenizer, aggregation_strategy="first", device=0)
mutations_tokenizer = AutoTokenizer.from_pretrained("Brizape/tmvar-PubMedBert-finetuned-24-02")
mutations_pipeline = pipeline("ner", model="Brizape/tmvar-PubMedBert-finetuned-24-02", tokenizer=mutations_tokenizer, aggregation_strategy="first", device=0)

def query_plain(text, url="http://localhost:8888/plain"):
    return requests.post(url, json={'text': text}).json()

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

class EntityRecognizer:
    def __init__(self, id_list, n_jobs):
        self.id_list = id_list
        self.n_jobs = n_jobs
        
    def _data_loader(self, id_list):
        to_concat = []
        for idx in id_list:
            df = pd.read_csv(INPUT_FILEPATH_CT + "%s_preprocessed.csv"%idx)
            to_concat.append(df)
        return to_concat
    
    def _mtner_normalize_format(self, json_data):
        spacy_format_entities = []
        for annotation in json_data["annotations"]:
            start = annotation["span"]["begin"]
            end = annotation["span"]["end"]
            label = annotation["obj"]
            mention = annotation["mention"]
            score = annotation["prob"]
            normalized_id = annotation["id"]
            spacy_format_entities.append({
                "entity_group": label,
                "text": mention,
                "score": score,
                "start": start,
                "end": end,
                "normalized_id": normalized_id
            })
        spacy_result = {
            "text": json_data["text"],
            "ents": spacy_format_entities,
        }
        return spacy_result
    
    def _merge_lists_with_priority_to_first(self, list1, list2):
        merged_list = list1.copy()  # Create a copy of list1 to preserve its contents
        
        for dict2 in list2:
            overlap = False
            for dict1 in list1:
                if (dict1['start'] <= dict2['end'] and dict2['start'] <= dict1['start']) or (dict2['start'] <= dict1['end'] and dict1['start'] <= dict2['start']):
                    overlap = True
                    break
            
            if not overlap:
                merged_list.append(dict2)
        return merged_list
    
    def _find_and_remove_overlaps(self, dictionary_list, if_overlap_keep):
        # Create a dictionary to store non-overlapping entries
        non_overlapping = {}
        # Create a set of entity groups to keep
        preferred_set = set(if_overlap_keep)

        # Iterate through the input list
        for entry in dictionary_list:
            text = entry['text']
            group = entry['entity_group']

            # Check if the text is already in the non_overlapping dictionary
            if text in non_overlapping:
                # Compare groups and keep the entry if it belongs to one of the preferred groups
                if group in preferred_set:
                    non_overlapping[text] = entry
            else:
                non_overlapping[text] = entry

        # Convert the non-overlapping dictionary back to a list
        result_list = list(non_overlapping.values())

        return result_list

    def negation_handling(self, sentence, entity):
        nlp = spacy.load("en_core_web_sm", disable={"ner"})
        nlp = medspacy.load(nlp)
        nlp.disable_pipe('medspacy_target_matcher')
        nlp.disable_pipe('medspacy_pyrush')
        @Language.component("add_custom_entity")
        def add_cutom_entity(doc):
            start_char = doc.text.find(entity["text"])
            end_char = start_char + len(entity["text"]) - 1  # Subtract 1 to get the inclusive end position
            start_token = None
            end_token = None
            # Find the corresponding tokens for the start and end positions
            for token in doc:
                if token.idx <= start_char < token.idx + len(token.text) and start_token is None:
                    start_token = token
                if token.idx <= end_char <= token.idx + len(token.text) and end_token is None:
                    end_token = token
                if start_token is not None and end_token is not None:
                    doc.set_ents([Span(doc, start_token.i, end_token.i + 1, entity["entity_group"])]) 
            return doc
        nlp.add_pipe("add_custom_entity", before='medspacy_context') 
        doc = nlp(sentence)
        for e in doc.ents:
            rs = str(e._.is_negated)
            if rs == "True": 
                entity["is_negated"] = "yes"
            else:
                entity["is_negated"] = "no"
        return  entity 
    
    def aberration_recognizer(self, text):
        med_nlp = medspacy.load()
        med_nlp.disable_pipe('medspacy_target_matcher')
        @Language.component("aberrations-ner")
        def regex_pattern_matcher_for_aberrations(doc):
            df_regex = pd.read_csv("../data/regex_variants.tsv", sep="\t", header=None)
            df_regex = df_regex.rename(columns={1 : "label", 2:"regex_pattern"}).drop(columns=[0])
            dict_regex = df_regex.set_index('label')['regex_pattern'].to_dict()
            original_ents = list(doc.ents)
            # Compile the regex patterns
            compiled_patterns = {
                label: re.compile(pattern)
                for label, pattern in dict_regex.items()
            }
            mwt_ents = []
            for label, pattern in compiled_patterns.items():
                for match in re.finditer(pattern, doc.text):
                    start, end = match.span()
                    span = doc.char_span(start, end)
                    if span is not None:
                        mwt_ents.append((label, span.start, span.end, span.text))
                        
            for ent in mwt_ents:
                label, start, end, name = ent
                per_ent = Span(doc, start, end, label=label)
                original_ents.append(per_ent)

            doc.ents = filter_spans(original_ents)
            
            return doc
        med_nlp.add_pipe("aberrations-ner", before='medspacy_context')
        doc = med_nlp(text)
        ent_list =[] 
        for entity in doc.ents:
            ent_list.append({"entity_group" : entity.label_, 
                            "text" : entity.text, 
                            "start": entity.start_char, 
                            "end": entity.end_char, 
                            "is_negated" : "yes" if entity._.is_negated else "no"})
        return ent_list
    
    def recognize_entities(self, df):
        nct_ids = []
        sentences = []
        entities_groups = []
        entities_texts = []
        normalized_ids = []
        is_negated = []
        criteria = []
        for _,row in df.iterrows():
            sent = row["sentence"]
            # print(sent)
            main_entities = self._mtner_normalize_format(query_plain(sent))["ents"]
            variants_entities = mutations_pipeline(sent)
            aberration_entities = self.aberration_recognizer(sent)
            aux_entities = aux_pipeline(sent)
            aux_entities = get_dictionaries_with_values(aux_entities, "entity_group", AUXILIARY_ENTITIES_LIST)
            aux_entities = [{"text" if k == "word" else k: v for k, v in d.items()} for d in aux_entities]
            # print(aux_entities)
            combined_entities = self._merge_lists_with_priority_to_first(variants_entities, main_entities)
            combined_entities  = self._merge_lists_with_priority_to_first(combined_entities, aux_entities)
            combined_entities = self._merge_lists_with_priority_to_first(combined_entities, aberration_entities)
            # print(combined_entities)
            # Convert the selected_entries dictionary back to a list
            if len(combined_entities) > 0:
                clean_entities = self._find_and_remove_overlaps(combined_entities, if_overlap_keep=["gene", "ProteinMutation", "DNAMutation", "SNP"])
                for ent in clean_entities:
                    if ("score" in ent and ent["score"] > 0.5) or ("score" not in ent):
                        ent = self.negation_handling(sent, ent)
                        is_negated.append(ent["is_negated"]) 
                        nct_ids.append(row["nct_id"])
                        sentences.append(sent)
                        entities_groups.append(ent['entity_group'])
                        entities_texts.append(ent['text'])
                        if "normalized_id" in ent:
                            normalized_ids.append(ent["normalized_id"])
                        else: 
                            normalized_ids.append("CUI-less")
                        criteria.append(row["criteria"])
                    else:
                            continue
        return pd.DataFrame({
                            'nct_id': nct_ids,
                            'sentence': sentences,
                            'entity_text': entities_texts,
                            'entity_group': entities_groups,
                            'normalized_id': normalized_ids,
                            'criteria' : criteria,
                            "is_negated" : is_negated
                        })
            
    def __call__(self):
        df = self._data_loader(self.id_list)
        parallel_runner = ParallelExecutor(n_jobs=self.n_jobs)(total=len(self.id_list))
        X = parallel_runner(
            joblib.delayed(self.recognize_entities)(
            ct_df, 
            )
            for ct_df in df
        )     
        all_trials = pd.concat(X)
        all_trials.to_csv(OUTPUT_FILEPATH_CT + "entities_parsed.csv")
        return all_trials