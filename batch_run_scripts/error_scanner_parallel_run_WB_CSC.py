# %% [markdown]
# ## Example script to run the Error-Scanner AI tool via API
# 
# This example retrieve metadata from the Metadata Editor and detects evidently incorrect, inconsistent, or contradictory information.
# 
# UI - https://w1lxscirender02.worldbank.org:8080/ai_for_data_playground

# %%
import os
import re
import json
import time
import math
import shutil
import tempfile
import requests
import threading
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from concurrent.futures import ThreadPoolExecutor, as_completed

# %%
def fetch_indicator_ids():
    url = "https://data360api.worldbank.org/data360/indicators"
    params = {
        "datasetId": "WB_CSC"
    }
    
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    
    return data

# %%
def fetch_indicator_metadata(indicator_id):
    base_metadata_url = "https://data360files.worldbank.org/data360-data/metadata/WB_CSC/{indicator_id}.json"
    url = base_metadata_url.format(indicator_id=indicator_id)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

# %%
# define a function to wait for a job and get the outputs
def wait_for_job_outputs(job):
    while job.done() != True:
        time.sleep(0.5)
    return job.outputs()

# %%
# define a function to extract JSON from the output text
def extract_json(text):
    idx = text.rfind("----------")
    text = text[idx:]
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)    
    if match:
        try:
            data = json.loads(match.group(1))
            return data
        except json.JSONDecodeError:
            return None
    return None

# %%
def atomic_dataframe_write(df, file_name):
    """
    Write a pandas dataframe to a CSV file in an atomic manner.
    """
    # Create a temporary file in the same directory as the target file
    dir_name = os.path.dirname(file_name)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dir_name, suffix='.xlsx') as tf:
        df.to_excel(tf.name, index=False)
        temp_file_name = tf.name
    # Atomically move the temporary file to the target file
    shutil.move(temp_file_name, file_name)

# %%
# create a Gradio client instance
gradio_client = Client("https://w1lxscirender02.worldbank.org:8080/ai_for_data_playground", ssl_verify=False)

# %%
def process_one(indicator_id):
    try:
        # create a new session
        job = gradio_client.submit(
            api_name="/error_scanner__create_session"
        )
        outputs = wait_for_job_outputs(job)
        session_id = outputs[0]

        # load agents manifest
        job = gradio_client.submit(
            file_name="error_scanner_kam_20251215.yml",
            session_id=session_id,
            api_name="/error_scanner__load_agents_manifest"
        )
        outputs = wait_for_job_outputs(job)
        agents_manifest = outputs[0][0]

        # create agents
        job = gradio_client.submit(
            agents_manifest=agents_manifest,
            gpt_model="gpt-5",
            session_id=session_id,
            api_name="/error_scanner__create_agents"
        )
        outputs = wait_for_job_outputs(job)
        
        # fetch metadata
        metadata = fetch_indicator_metadata(indicator_id)
        metadata_to_scan = metadata["series_description"]
        if "ref_country" in metadata_to_scan:
            del metadata_to_scan["ref_country"]
        if "geographic_units" in metadata_to_scan:
            del metadata_to_scan["geographic_units"]
        indicator_name = metadata_to_scan["name"]

        # start agents activity
        job = gradio_client.submit(
            metadata_to_scan=metadata_to_scan,
            session_id=session_id,
            api_name="/error_scanner__start_agents_activity",
        )
        outputs = wait_for_job_outputs(job)

        # Parse detected issues into pretty JSON array text
        issues_list = extract_json(outputs[-1][0])
        json_text = "[\n    " + ",\n    ".join(
            json.dumps(obj, ensure_ascii=False) for obj in issues_list
        ) + "\n]"

        # delete the session
        job = gradio_client.submit(
            session_id=session_id,
            api_name="/error_scanner__delete_session"
        )
        outputs = wait_for_job_outputs(job)

        return {
            "indicator_id": indicator_id,
            "indicator_name": indicator_name,
            "detected_issues": json_text,
        }

    except Exception as e:
        print(f"[Warning] Failed for indicator: {indicator_id}. Reason: {e}")
        return None

# %%
MAX_WORKERS = 10
def run_parallel(todo_items, output_df, output_file_name):
    results = output_df.to_dict(orient="records")

    # skip projects that are already completed
    done = set(output_df["indicator_id"].astype(str))
    todo = [indicator_id for indicator_id in todo_items if indicator_id not in done]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one, indicator_id) for indicator_id in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            res = fut.result()
            if res is not None:
                results.append(res)

                df_results = pd.DataFrame(results) if results else pd.DataFrame(
                    columns=['indicator_id', 'indicator_name', 'detected_issues']
                )

                atomic_dataframe_write(df_results, output_file_name)
                

# %%
todo = fetch_indicator_ids()

# %%
output_file_name = "WB_CSC_detected_metadata_issues_20260206.xlsx"
if os.path.exists(output_file_name):
    output_df = pd.read_excel(output_file_name)
else:
    column_names = ['indicator_id', 'indicator_name', 'detected_issues']
    output_df = pd.DataFrame(columns=column_names)

# %%
run_parallel(todo, output_df, output_file_name)

# %%


# %%



