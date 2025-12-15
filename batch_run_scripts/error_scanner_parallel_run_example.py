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
load_dotenv()
ME_API_KEY = os.getenv("ME_API_KEY")
ME_URL = 'https://metadataeditor.worldbank.org/index.php'
ME_HEADERS = {"X-API-KEY": ME_API_KEY}

# %%
# define functions for Metadata Editor operations
def fetch_me_collection_list():
    """
    Fetch collection list from the Metadata Editor.
    """
    response = requests.get(f"{ME_URL}/api/collections", headers=ME_HEADERS)
    response.raise_for_status()
    collection_list = [f"[{collection['id']}] {collection['title']}" for collection in response.json()['collections']]
    collection_list = sorted(collection_list, key=lambda s: int(re.search(r"\[(\d+)\]", s).group(1)) if re.search(r"\[(\d+)\]", s) else float("inf"))
    return collection_list

def fetch_me_project_list(collection):
    """
    Fetch project list from the Metadata Editor.
    """
    collection_id = re.search(r"\[(\d+)\]", collection).group(1)
    search_params = []
    search_params.append(f"collection={collection_id}")   
    probe_params = search_params.copy()
    probe_params.append(f"offset=0&limit=1")
    response = requests.get(f"{ME_URL}/api/editor/?{'&'.join(probe_params)}", headers=ME_HEADERS)
    total_cases = response.json().get("total", 0)
    limit = 500
    project_list = []
    num_pages = math.ceil(total_cases / limit) if limit else 0
    for offset in tqdm(range(0, total_cases, limit), total=num_pages):
        search_more_params = search_params.copy()
        search_more_params.append(f"offset={offset}&limit={limit}")
        response = requests.get(f"{ME_URL}/api/editor/?{'&'.join(search_more_params)}", headers=ME_HEADERS)
        if response.status_code != 200:
            raise Exception(f"Something wrong with the Metadata Editor search: {response.text}")
        data = response.json()
        project_list.extend(data.get("projects", []))

    project_title_list = [f"[{project['id']}] {project['title']}" for project in project_list]
    project_title_list = sorted(project_title_list, key=lambda s: int(re.search(r"\[(\d+)\]", s).group(1)) if re.search(r"\[(\d+)\]", s) else float("inf"))
    default_value = project_title_list[0] if project_title_list else None
    return project_title_list

def fetch_me_project_metadata(project):
    """
    Fetch project metadata from the Metadata Editor.
    """
    project_id = re.search(r"\[(\d+)\]", project).group(1)
    response = requests.get(f"{ME_URL}/api/editor/{project_id}", headers=ME_HEADERS)
    if response.status_code != 200:
        raise Exception(f"Something wrong with the Metadata Editor search: {response.text}")
    metadata = response.json()['project']['metadata']
    metadata.get("series_description", {}).pop("ref_country", None)
    metadata.get("series_description", {}).pop("geographic_units", None)
    return metadata

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
def process_one(me_collection, me_project):
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
            gpt_model="gpt-5-mini",
            session_id=session_id,
            api_name="/error_scanner__create_agents"
        )
        outputs = wait_for_job_outputs(job)
        
        # fetch metadata
        metadata_to_scan = fetch_me_project_metadata(me_project)

        # start agents activity
        job = gradio_client.submit(
            metadata_to_scan=metadata_to_scan,
            session_id=session_id,
            api_name="/error_scanner__start_agents_activity",
        )
        outputs = wait_for_job_outputs(job)

        # Build URL from [12345] pattern in me_project
        m = re.search(r"\[(\d+)\]", str(me_project))
        if not m:
            raise ValueError("Could not extract project_id from ME_project (expected [12345] pattern).")
        project_id = m.group(1)
        me_url = f"https://metadataeditor.worldbank.org/index.php/api/editor/{project_id}"

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
            "ME_collection": me_collection,
            "ME_project": me_project,
            "ME_url": me_url,
            "detected_issues": json_text,
        }

    except Exception as e:
        print(f"[Warning] Failed for project: {me_project}. Reason: {e}")
        return None

# %%
MAX_WORKERS = 10
def run_parallel(todo_items, output_df, output_file_name):
    results = output_df.to_dict(orient="records")

    # skip projects that are already completed
    done = set(output_df["ME_project"].astype(str))
    todo = [(collection, project) for (collection, project) in todo_items if str(project) not in done]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one, collection, project) for (collection, project) in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            res = fut.result()
            if res is not None:
                results.append(res)

                df_results = pd.DataFrame(results) if results else pd.DataFrame(
                    columns=["ME_collection", "ME_project", "ME_url", "detected_issues"]
                )

                atomic_dataframe_write(df_results, output_file_name)
                

# %%
me_collection = "[1] World Development Indicators"
me_project_list = fetch_me_project_list(me_collection)

# %%
todo = [(me_collection, me_project) for me_project in me_project_list]

# %%
output_file_name = "WDI_metadata_issues_20251215.xlsx"
if os.path.exists(output_file_name):
    output_df = pd.read_excel(output_file_name)
else:
    column_names = ['ME_collection', 'ME_project', 'ME_url', 'detected_issues']
    output_df = pd.DataFrame(columns=column_names)

# %%
run_parallel(todo, output_df, output_file_name)

# %%


# %%



