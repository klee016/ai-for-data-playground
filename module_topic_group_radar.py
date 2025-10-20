import re
import os
import time
import json
import yaml
import math
import logging
import requests
import numpy as np
import pandas as pd
import gradio as gr
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import plotly.graph_objects as go

import httpx
http_client = httpx.Client(verify=False)

# Set up logging format and level for debugging and monitoring
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logging.getLogger("autogen_core").setLevel(logging.WARNING)

AGENTS_MANIFEST_DIR = "assets_topic_group_radar/agents_manifest"
AGENTS_MANIFEST_DEFAULT_FILE = "default_agents_manifest.yml"
TOPIC_GROUP_TAXONOMY_DIR = "assets_topic_group_radar/topic_group_taxonomy"
TOPIC_GROUP_TAXONOMY_DEFAULT_FILE = "default_topic_group_taxonomy.yml"

class TopicGroupRadar:
    def __init__(self, openai_api_key, me_api_key):
        """
        Initialize the ErrorScanner object.
        """
        logging.info("Initializing ErrorScanner object...")
        self.openai_api_key = openai_api_key
        self.me_url = 'https://metadataeditor.worldbank.org/index.php'
        self.me_headers = {"X-API-KEY": me_api_key}
        self.embedding_model = "text-embedding-3-small"
        self.gpt_model = "gpt-4.1-mini"
        self.gpt_model_client = None
        self.logit_bias = []
        self.topic_group_list = []
    
    def create_gpt_model_client(self, gpt_model):
        """
        Create an OpenAI model client.
        """
        logging.info("Creating an OpenAI model client...")
        self.gpt_model = gpt_model
        self.gpt_model_client = OpenAI(api_key=self.openai_api_key, http_client=http_client)
        tokenizer = tiktoken.encoding_for_model(self.gpt_model)
        self.logit_bias = [tokenizer.encode(token)[0] for token in ["Yes", "No"]]

    def refresh_agents_manifest_files_dd(self):
        """
        Refresh the list of agents manifest YAML files.
        """
        logging.info("Refreshing the list of agents manifest YAML files...")
        files = [f for f in os.listdir(AGENTS_MANIFEST_DIR) if os.path.isfile(os.path.join(AGENTS_MANIFEST_DIR, f))]
        files.sort()
        return gr.update(choices=files)

    def load_agents_manifest_file(self, file_name):
        """
        Load agents manifest YAML file.
        """
        logging.info("Loading agents manifest YAML file...")
        print(file_name)
        try:
            with open(os.path.join(AGENTS_MANIFEST_DIR, file_name), "r", encoding="utf-8") as f:
                return f.read(), gr.update(value=file_name)
        except FileNotFoundError:
            return "", None

    def save_agents_manifest_file(self, content, file_name):
        """
        Save agents manifest YAML file.
        """
        logging.info("Saving agents manifest YAML file...")
        with open(os.path.join(AGENTS_MANIFEST_DIR, file_name), "w", encoding="utf-8") as f:
            f.write(content)
        gr.Info("Agents manifest saved successfully!")

    def delete_agents_manifest_file(self, file_name):
        """
        Delete agents manifest YAML file.
        """
        logging.info("Deleting agents manifest YAML file...")
        os.remove(os.path.join(AGENTS_MANIFEST_DIR, file_name))
        gr.Info("Agents manifest deleted successfully!")

    def create_agents(self, agents_manifest, embedding_model, gpt_model):
        """
        Create agents according to the agents manifest YAML file.
        """
        logging.info("Creating agents according to the agents manifest YAML file...")
        manifest = yaml.safe_load(agents_manifest)
        self.embedding_model = embedding_model
        self.create_gpt_model_client(gpt_model)
        self.agent_list = []
        for entry in manifest.get("agents_manifest", []):
            name = entry.get("name")
            system_message = entry.get("system_message")
            if not name or not system_message:
                print(f"Skipping invalid agent entry: {entry}")
                continue
            agent = {
                "name": name,
                "system_message": system_message
            }
            self.agent_list.append(agent) 
        
        gr.Info("Agents created successfully!")
        return gr.update()
    
    def refresh_topic_group_taxonomy_files_dd(self):
        """
        Refresh the list of topic group taxonomy YAML files.
        """
        logging.info("Refreshing the list of topic group taxonomy YAML files...")
        files = [f for f in os.listdir(TOPIC_GROUP_TAXONOMY_DIR) if os.path.isfile(os.path.join(TOPIC_GROUP_TAXONOMY_DIR, f))]
        files.sort()
        return gr.update(choices=files)

    def load_topic_group_taxonomy_file(self, file_name):
        """
        Load topic group taxonomy YAML file.
        """
        logging.info("Loading topic group taxonomy YAML file...")
        try:
            with open(os.path.join(TOPIC_GROUP_TAXONOMY_DIR, file_name), "r", encoding="utf-8") as f:
                return f.read(), gr.update(value=file_name)
        except FileNotFoundError:
            return "", None

    def save_topic_group_taxonomy_file(self, content, file_name):
        """
        Save topic group taxonomy YAML file.
        """
        logging.info("Saving topic group taxonomy YAML file...")
        with open(os.path.join(TOPIC_GROUP_TAXONOMY_DIR, file_name), "w", encoding="utf-8") as f:
            f.write(content)
        gr.Info("Topic group taxonomy saved successfully!")

    def delete_topic_group_taxonomy_file(self, file_name):
        """
        Delete agents manifest YAML file.
        """
        logging.info("Deleting topic group taxonomy YAML file...")
        os.remove(os.path.join(TOPIC_GROUP_TAXONOMY_DIR, file_name))
        gr.Info("Topic group taxonomy deleted successfully!")
    
    def safe_text(self, x):
        if x is None:
            return ""
        if isinstance(x, (dict, list)):
            return json.dumps(x, ensure_ascii=False, sort_keys=False)
        return str(x)

    def get_embedding(self, text: str, model: str):
        text = text.replace("\n", " ")
        embedding = self.gpt_model_client.embeddings.create(input=[text], model=model).data[0].embedding
        return np.array(embedding, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    
    def create_topic_groups(self, topic_group_taxonomy):
        """
        Create topic groups according to the topic group taxonomy YAML file.
        """
        logging.info("Creating topic groups according to the topic group taxonomy YAML file...")
        taxonomy = yaml.safe_load(topic_group_taxonomy)

        self.topic_group_list = []
        for entry in taxonomy.get("topic_group_taxonomy", []):
            name = entry.get("name")
            description = entry.get("description")
            if not name or not description:
                print(f"Skipping invalid topic group entry: {entry}")
                continue
            topic_group = {
                "name": name,
                "description": description,
                "embedding": self.get_embedding(self.safe_text(description), self.embedding_model)
            }
            self.topic_group_list.append(topic_group) 
        
        gr.Info("Topic groups created successfully!")
        return gr.update()

    def fetch_me_collection_list(self):
        """
        Fetch collection list from the Metadata Editor.
        """
        logging.info("Fetching collection list from the Metadata Editor...")
        try:
            response = requests.get(f"{self.me_url}/api/collections", headers=self.me_headers)
            response.raise_for_status()
            collection_list = [f"[{collection['id']}] {collection['title']}" for collection in response.json()['collections']]
            collection_list = sorted(collection_list, key=lambda s: int(re.search(r"\[(\d+)\]", s).group(1)) if re.search(r"\[(\d+)\]", s) else float("inf"))
            return gr.update(choices=collection_list, value="[5] WDI - Environment")
        except Exception as e:
            return gr.update(choices=[], value=None), f"Error: {e}"

    def fetch_me_project_list(self, collection, progress=gr.Progress()):
        """
        Fetch project list from the Metadata Editor.
        """
        logging.info("Fetching project list from the Metadata Editor...")
        collection_id = re.search(r"\[(\d+)\]", collection).group(1)
        
        search_params = []
        # search_params.append(f"type=timeseries-db")
        search_params.append(f"collection={collection_id}")   

        probe_params = search_params.copy()
        probe_params.append(f"offset=0&limit=1")
        response = requests.get(f"{self.me_url}/api/editor/?{'&'.join(probe_params)}", headers=self.me_headers)
        total_cases = response.json().get("total", 0)

        limit = 500
        project_list = []
        num_pages = math.ceil(total_cases / limit) if limit else 0
        for offset in progress.tqdm(range(0, total_cases, limit), total=num_pages):
            search_more_params = search_params.copy()
            search_more_params.append(f"offset={offset}&limit={limit}")
            response = requests.get(f"{self.me_url}/api/editor/?{'&'.join(search_more_params)}", headers=self.me_headers)
            if response.status_code != 200:
                raise gr.Error(f"Something wrong with the Metadata Editor search: {response.text}")
    
            data = response.json()
            project_list.extend(data.get("projects", []))

        project_title_list = [f"[{project['id']}] {project['title']}" for project in project_list]
        project_title_list = sorted(project_title_list, key=lambda s: int(re.search(r"\[(\d+)\]", s).group(1)) if re.search(r"\[(\d+)\]", s) else float("inf"))
        default_value = project_title_list[0] if project_title_list else None
        return gr.update(choices=project_title_list, value=default_value)

    def fetch_me_project_metadata(self, project):
        """
        Fetch project metadata from the Metadata Editor.
        """
        logging.info("Fetching project metadata from the Metadata Editor...")
        project_id = re.search(r"\[(\d+)\]", project).group(1)
        
        response = requests.get(f"{self.me_url}/api/editor/{project_id}", headers=self.me_headers)

        if response.status_code != 200:
            raise gr.Error(f"Something wrong with the Metadata Editor search: {response.text}")
        metadata = response.json()['project']['metadata']
        metadata.get("series_description", {}).pop("ref_country", None)
        metadata.get("series_description", {}).pop("geographic_units", None)
        return metadata

    def get_cross_encoding(self, metadata, topic_group):
        """
        Get cross-encoding score with OpenAI GPT.
        """          
        system_message = self.agent_list[0]['system_message'] 
        user_message = '''
            Metadata: {metadata}
            Topic Group: {topic_group}
            Relevant:
        '''
        
        response = self.gpt_model_client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message.format(metadata=metadata, topic_group=topic_group)}
            ],
            temperature=0,
            logprobs=True,
            logit_bias={self.logit_bias[0]: 1, self.logit_bias[1]: 1},
        )

        return (
            response.choices[0].message.content,
            response.choices[0].logprobs.content[0].logprob,
        )

    def calculate_semantic_similarities(self, metadata_to_calculate):
        metadata_all = metadata_to_calculate
        metadata_selected = {
            "name": metadata_all.get("series_description", {}).get("name", ""),
            "definition_long": metadata_all.get("series_description", {}).get("definition_long", ""),
            # "methodology": metadata_all.get("series_description", {}).get("methodology", ""),
            # "dimensions": metadata_all.get("series_description", {}).get("dimensions", "")
        }        
        metadata_embedding = self.get_embedding(self.safe_text(metadata_selected), self.embedding_model) 
        
        # baseline_text = "Education; Gender; Health; Nutrition & Population; Social Protection; Economic Policy; Finance; Institutions; Poverty; Trader, Investment and Competitiveness; Agriculture and Food; Climate Change; Environment; Social Sustainability; Water; Energy & Extractives; Global Infrastructure Finance; Transport; Urban, Resilience and Land; Data Infrastructure; Cybersecurity; Digital Industry and Jobs; Digital Services"
        baseline_text = "This document outlines key objectives, methodologies, and expected outcomes of the proposed activities. It aims to support coordination among relevant stakeholders and ensure alignment with institutional priorities."
        baseline_embedding = self.get_embedding(baseline_text, self.embedding_model)
        baseline_cosine_similarity = self.cosine_similarity(metadata_embedding, baseline_embedding)
        print(f"baseline_cosine_similarity: {baseline_cosine_similarity}")

        topic_group_names = []
        topic_group_descriptions = []
        cosine_similarities = []
        cross_encoding_scores = []
        for topic_group in self.topic_group_list:
            topic_group_names.append(str(topic_group.get("name", "")).strip() or "unnamed")
            topic_group_descriptions.append(str(topic_group.get("description", "")).strip() or "none")
            topic_group_embedding = topic_group.get("embedding", [])
            cosine_similarity = self.cosine_similarity(metadata_embedding, topic_group_embedding) 
            cosine_similarities.append(cosine_similarity)
            cross_encoding_prediction, cross_encoding_logprob = self.get_cross_encoding(self.safe_text(metadata_selected), str(topic_group.get("description", "")).strip())
            cross_encoding_probability  = math.exp(cross_encoding_logprob)
            cross_encoding_score = cross_encoding_probability * -1 + 1 if cross_encoding_prediction == "No" else cross_encoding_probability
            cross_encoding_scores.append(cross_encoding_score)       

        return topic_group_names, cosine_similarities, cross_encoding_scores, baseline_cosine_similarity, None
        
    def draw_radar_charts(self, topic_group_names, cosine_similarities, cross_encoding_scores, baseline_cosine_similarity):
        theta = topic_group_names + [topic_group_names[0]] if topic_group_names else []
        r_cosine_similarities = cosine_similarities + [cosine_similarities[0]] if cosine_similarities else []
        r_cross_encoding_scores = cross_encoding_scores + [cross_encoding_scores[0]] if cross_encoding_scores else []
        r_cosine_similarities = np.array(r_cosine_similarities, dtype=float)
        r_cross_encoding_scores = np.array(r_cross_encoding_scores, dtype=float)
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=r_cosine_similarities, theta=theta, fill="toself", name="cosine similarity", subplot="polar1"))
        fig.add_trace(go.Scatterpolar(r=r_cross_encoding_scores, theta=theta, fill="toself", name="cross-encoding score", subplot="polar2"))
        fig.update_layout(
            title="Semantic Similarity (metadata vs topics)",
            showlegend=True,
            margin=dict(l=20, r=20, t=50, b=20),
            polar1=dict(
                radialaxis=dict(
                    visible=True,
                    range=[float(baseline_cosine_similarity), 0.4]
                )
            ),
            polar2=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-0.1, 1.1] 
                )
            ),
        )
        return fig
    
    def get_additional_info(self, metadata):
        name = metadata.get("series_description", {}).get("name", "")
        definition_long = metadata.get("series_description", {}).get("definition_long", "")
        name_embedding = self.get_embedding(self.safe_text(name), self.embedding_model)
        definition_embedding = self.get_embedding(self.safe_text(definition_long), self.embedding_model)
        cosine_similarity = self.cosine_similarity(name_embedding, definition_embedding)
        return_obj = {
            "name": name,
            "definition_long": definition_long,
            "cosine_similarity": cosine_similarity
        }
        return return_obj
    
    def handler(self):
        """
        Set up the Gradio Blocks UI for interacting with the search system.
        
        Returns:
            gr.Blocks: The Gradio interface definition.
        """
        with gr.Blocks().queue(max_size=50, default_concurrency_limit=50) as handler:

            # UI section for agent description
            gr.Markdown(
                """
                ### The Topic-Group-Radar numerically estimates how closely a dataset’s metadata aligns with each topic group description using cosine similarity and cross-encoding, treating dataset classification as a regression problem rather than categorical labeling.
                """
            )

            # UI section for selecting GPT model
            guide_md1 = gr.Markdown(
                """
                ### 1️⃣ Select an embedding model (cosine similarity) and a GPT model (cross-embedding score).
                """
            )
            embedding_model_rdo = gr.Radio(["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"], value=self.embedding_model, label="Embedding models")
            gpt_model_rdo = gr.Radio(["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"], value=self.gpt_model, label="GPT models")


            # UI section for defining agents
            guide_md3 = gr.Markdown(
                """
                ### 2️⃣ Define your agents in YAML format.
                """
            )
            with gr.Row():
                agents_manifest_files_dd = gr.Dropdown(choices=None, container=False, value=AGENTS_MANIFEST_DEFAULT_FILE, interactive=True, allow_custom_value=True)
                load_agents_manifest_file_btn = gr.Button("Load Agents Manifest", variant="primary")
            agents_manifest_tb = gr.Textbox(value=None, lines=15, max_lines=15, label="Add/Edit agents manifest")
            with gr.Row():
                agents_manifest_file_name_tb = gr.Textbox(value=None, container=False, max_lines=1, interactive=True)
                save_agents_manifest_file_btn = gr.Button("Save File As")
                delete_agents_manifest_file_btn = gr.Button("Delete File")
            create_agents_btn = gr.Button("Create agents", variant="primary")

            
            # UI section for defining topic groups
            guide_md2 = gr.Markdown(
                """
                ### 3️⃣ Define your topic group taxonomy in YAML format.
                """
            )
            with gr.Row():
                topic_group_taxonomy_files_dd = gr.Dropdown(choices=None, container=False, value=TOPIC_GROUP_TAXONOMY_DEFAULT_FILE, interactive=True, allow_custom_value=True)
                load_topic_group_taxonomy_file_btn = gr.Button("Load Topic Group Taxonomy", variant="primary")
            topic_group_taxonomy_tb = gr.Textbox(value=None, lines=15, max_lines=15, label="Define/Edit topic group taxonomy")
            with gr.Row():
                topic_group_taxonomy_file_name_tb = gr.Textbox(value=None, container=False, max_lines=1, interactive=True)
                save_topic_group_taxonomy_file_btn = gr.Button("Save File As")
                delete_topic_group_taxonomy_file_btn = gr.Button("Delete File")
            create_topic_groups_btn = gr.Button("Create topic groups", variant="primary")

            
            # UI section for selecting an indicator
            guide_md4 = gr.Markdown(
                """
                ### 4️⃣ Select a project from the Metadata Editor          
                """
            )
            me_collection_dd = gr.Dropdown(choices=[], label="Select a collection", value=None, interactive=True)
            me_project_dd = gr.Dropdown(choices=[], label="Select a project", value=None, interactive=True, allow_custom_value=True)


            # UI section for calculating semantic similarities
            guide_md5 = gr.Markdown(
                """
                ### 5️⃣ Calculate semantic similarities to topic groups           
                """
            )
            calculate_semantic_similarities_btn = gr.Button("Calculate Semantic Similarities", variant="primary")
            metadata_to_calculate_js = gr.JSON(visible=False)
            topic_group_names_js = gr.JSON(visible=False)
            cosine_similarities_js = gr.JSON(visible=False)
            cross_encoding_scores_js = gr.JSON(visible=False)
            baseline_cosine_similarity_js = gr.JSON(visible=False)
            topic_group_radar_plt = gr.Plot(label="Topic Group Radar Chart")
            additional_info_js = gr.JSON(label="Additional Info")

            # Actions
            load_agents_manifest_file_btn.click(
                fn=self.load_agents_manifest_file, 
                inputs=[agents_manifest_files_dd], 
                outputs=[agents_manifest_tb, agents_manifest_file_name_tb], 
                show_api=True, api_name="topic_group_radar__load_agents_manifest")
            save_agents_manifest_file_btn.click(
                fn=self.save_agents_manifest_file, 
                inputs=[agents_manifest_tb, agents_manifest_file_name_tb], 
                show_api=False
            ).then(
                fn=self.refresh_agents_manifest_files_dd,
                inputs=None,
                outputs=[agents_manifest_files_dd],
                show_api=False
            )
            delete_agents_manifest_file_btn.click(
                fn=self.delete_agents_manifest_file, 
                inputs=[agents_manifest_file_name_tb], 
                show_api=False
            ).then(
                fn=self.refresh_agents_manifest_files_dd,
                inputs=None,
                outputs=[agents_manifest_files_dd],
                show_api=False
            )
            create_agents_btn.click(
                fn=self.create_agents, 
                inputs=[agents_manifest_tb, embedding_model_rdo, gpt_model_rdo], 
                outputs=[agents_manifest_tb], 
                show_api=True, api_name="topic_group_radar__create_agents"
            )
            load_topic_group_taxonomy_file_btn.click(
                fn=self.load_topic_group_taxonomy_file, 
                inputs=[topic_group_taxonomy_files_dd], 
                outputs=[topic_group_taxonomy_tb, topic_group_taxonomy_file_name_tb], 
                show_api=True, api_name="topic_group_radar__load_topic_group_taxonomy"
            )
            save_topic_group_taxonomy_file_btn.click(
                fn=self.save_topic_group_taxonomy_file, 
                inputs=[topic_group_taxonomy_tb, topic_group_taxonomy_file_name_tb],
                show_api=False
            ).then(
                fn=self.refresh_topic_group_taxonomy_files_dd,
                inputs=None,
                outputs=[topic_group_taxonomy_files_dd],
                show_api=False
            )
            delete_topic_group_taxonomy_file_btn.click(
                fn=self.delete_topic_group_taxonomy_file, 
                inputs=[topic_group_taxonomy_file_name_tb], 
                show_api=False
            ).then(
                fn=self.refresh_topic_group_taxonomy_files_dd,
                inputs=None,
                outputs=[topic_group_taxonomy_files_dd],
                show_api=False
            )
            create_topic_groups_btn.click(
                fn=self.create_topic_groups, 
                inputs=[topic_group_taxonomy_tb], 
                outputs=[topic_group_taxonomy_tb], 
                show_api=True, api_name="topic_group_radar__create_topic_groups"
            )
            handler.load(self.fetch_me_collection_list, inputs=None, outputs=[me_collection_dd], show_api=False)
            handler.load(self.refresh_agents_manifest_files_dd, inputs=None, outputs=[agents_manifest_files_dd], show_api=False)
            handler.load(self.refresh_topic_group_taxonomy_files_dd, inputs=None, outputs=[topic_group_taxonomy_files_dd], show_api=False)
            me_collection_dd.change(fn=self.fetch_me_project_list, inputs=[me_collection_dd], outputs=[me_project_dd], show_api=False)
            calculate_semantic_similarities_btn.click(
                fn=self.fetch_me_project_metadata,
                inputs=[me_project_dd],
                outputs=[metadata_to_calculate_js],
                show_api=False
            ).then(
                fn=self.calculate_semantic_similarities, 
                inputs=[metadata_to_calculate_js], 
                outputs=[topic_group_names_js, cosine_similarities_js, cross_encoding_scores_js, baseline_cosine_similarity_js, topic_group_radar_plt], 
                show_api=True, api_name="topic_group_radar__calculate_semantic_similarities"
            ).then(
                fn=self.draw_radar_charts,
                inputs=[topic_group_names_js, cosine_similarities_js, cross_encoding_scores_js, baseline_cosine_similarity_js],
                outputs=[topic_group_radar_plt],
                show_api=False
            ).then(
                fn=self.get_additional_info,
                inputs=[metadata_to_calculate_js],
                outputs=[additional_info_js],
                show_api=True, api_name="topic_group_radar__get_additional_info"
            )
        
        return handler