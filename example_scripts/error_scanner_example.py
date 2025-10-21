# %% [markdown]
# ## Example script to run the Error-Scanner AI tool via API
# 
# This example retrieve metadata from the WDI – Environment collection and detects evidently incorrect, inconsistent, or contradictory information.
# 
# UI - https://w1lxscirender02.worldbank.org:8082/ai_for_data_playground

# %%
import os
import re
import json
import time
import math
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from gradio_client import Client, handle_file

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
# create a Gradio client instance
gradio_client = Client("https://w1lxscirender02.worldbank.org:8082/ai_for_data_playground", ssl_verify=False)

# %%
# create a new session
job = gradio_client.submit(
	api_name="/error_scanner__create_session"
)
outputs = wait_for_job_outputs(job)
session_id = outputs[0]

# %%
# load agents manifest
job = gradio_client.submit(
	file_name="default_agents_manifest.yml",
	session_id=session_id,
	api_name="/error_scanner__load_agents_manifest"
)
outputs = wait_for_job_outputs(job)
agents_manifest = outputs[0][0]

# %%
# create agents
job = gradio_client.submit(
	agents_manifest=agents_manifest,
	gpt_model="gpt-5-mini",
	session_id=session_id,
	api_name="/error_scanner__create_agents"
)
outputs = wait_for_job_outputs(job)

# %%
me_collection = "[5] WDI - Environment"
me_project_list = fetch_me_project_list(me_collection)

# %%
output_file_name = "WDI_Environment_detected_metadata_issues.xlsx"

if os.path.exists(output_file_name):
    df_detected_issues = pd.read_excel(output_file_name)
else:
    column_names = ['ME_collection', 'ME_project', 'ME_url', 'detected_issues']
    df_detected_issues = pd.DataFrame(columns=column_names)

for me_project in tqdm(me_project_list):
    if (df_detected_issues['ME_project'] == me_project).any():
        continue

    # fetch metadata
    metadata_to_scan = fetch_me_project_metadata(me_project)

    # start agents activity
    job = gradio_client.submit(
        metadata_to_scan=metadata_to_scan,
        session_id=session_id,
        api_name="/error_scanner__start_agents_activity"
    )
    outputs = wait_for_job_outputs(job)

    try:
        # record the detected issue
        project_id = re.search(r"\[(\d+)\]", me_project).group(1)
        me_url = f"https://metadataeditor.worldbank.org/index.php/api/editor/{project_id}"
        json_text = "[\n    " + ",\n    ".join(json.dumps(obj, ensure_ascii=False) for obj in extract_json(outputs[-1][0])) + "\n]"
        df_detected_issues.loc[len(df_detected_issues)] = [
            me_collection,
            me_project,
            me_url,
            json_text
        ]
        df_detected_issues.to_excel(output_file_name, index=False)
    except Exception as e:
        print(f"[Warning] Failed to record detected issue for project: {me_project}. Reason: {e}")
        continue


# %%



