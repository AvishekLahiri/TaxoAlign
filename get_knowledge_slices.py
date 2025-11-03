
import random
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse 
import yaml 
import shutil
import pickle
from datetime import datetime 
from tqdm import tqdm

import torch

#from pyserini.search import FaissSearcher
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import get_knowledge_slices_prompt

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def get_model_response(tokenizer, model, prompt):
    #tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
    #model = AutoModelForCausalLM.from_pretrained(
    #    model_path,
    #    device_map="auto",
    #    torch_dtype='auto',
    #    token="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"
    #    )

    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'), pad_token_id=tokenizer.eos_token_id, max_new_tokens=1024)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    print(response)
    return response


def get_knowledge_slices(tokenizer, model, documents, topic):
    responses = []
    for d in tqdm(documents):
        text = d[0] +'\n'+ d[1]
        knowledge_slices_prompt = get_knowledge_slices_prompt(text, topic)
        
        messages = [{"role": "user", "content": knowledge_slices_prompt}]

        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'), pad_token_id=tokenizer.eos_token_id, max_new_tokens=1024)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        responses.append(response)
    return responses
    
def get_knowledge_slices_llama(tokenizer, model, documents, topic):
    responses = []
    for d in tqdm(documents):
        text = d[0] +'\n'+ d[1]
        knowledge_slices_prompt = get_knowledge_slices_prompt(text, topic)
        
        messages = [{"role": "user", "content": knowledge_slices_prompt}]

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = model.generate(input_ids, max_new_tokens=1024, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9,)
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        responses.append(response)
    return responses

if __name__ == "__main__":

    set_seed(42)

    model_name = "mistralai/Mistral-7B-Instruct-v0.3" #"meta-llama/Meta-Llama-3-8B-Instruct" #"allenai/Llama-3.1-Tulu-3-8B" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token="your_token_here"
            )  

    with open('final_ref_and_cited_papers_dict.pkl', 'rb') as f:
        cited_papers_dict = pickle.load(f)
    with open('initial_selected_papers_title.pkl', 'rb') as f:
        initial_selected_papers_title = pickle.load(f)
    print("Number of survey papers: ", len(cited_papers_dict))

    keyphrases = {}
    annotated= ["3439723"]
    ctr=0

    for i in cited_papers_dict:
        
        keyphrases = {}
        
        fname = i[:i.rfind(".")]
        paper_name = initial_selected_papers_title[fname][0].replace('\n', '')
        paper_abstract = initial_selected_papers_title[fname][1].replace('\n', '')

        #if fname not in annotated:
        #    continue
        ctr=ctr+1
        if ctr<100:
            continue

        print(fname, paper_name)

        references = cited_papers_dict[i]
        documents = []
        for j in references:
            s2_result = references[j]
            try:
                if type(s2_result) is not str:
                    if s2_result['abstract'] is not None:
                        title = s2_result['title']
                        abstract = s2_result['abstract']
                        documents.append([title, abstract])
            except:
                continue
        
        responses = get_knowledge_slices(tokenizer, model, documents, paper_name)
        #keyphrase_responses = ["Doc1", "Doc2", "Doc3"]

        keyphrases[i] = responses

    with open("knowledge_slices/knowledge_slices_mistral"+fname+".pkl", 'wb') as f:
            pickle.dump(keyphrases, f)
