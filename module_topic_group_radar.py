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

# Set up logging format and level for debugging and monitoring
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logging.getLogger("autogen_core").setLevel(logging.WARNING)

AGENTS_MANIFEST_DIR = "agents_manifest_toic_group_radar"
AGENTS_MANIFEST_EXTENSION = ".yml"
REFERENCE_DIR = "reference_topic_group_radar"
REFERENCE_EXTENSION = ".yml"

class TopicGroupRadar:
    def __init__(self, openai_api_key, me_api_key):
        """
        Initialize the ErrorScanner object.
        """
        logging.info("Initializing ErrorScanner object...")
        self.openai_api_key = openai_api_key
        self.me_url = 'https://metadataeditor.worldbank.org/index.php'
        self.me_headers = {"X-API-KEY": me_api_key}
        self.model_client = None
        self.create_model_client()
        self.topic_group_list = []
    
    def create_model_client(self):
        """
        Create an OpenAI model client.
        """
        logging.info("Creating an OpenAI model client...")
        self.model_client = OpenAI(api_key=self.openai_api_key)

        print("token IDs for Yes and No")
        tokenizer = tiktoken.encoding_for_model("gpt-4.1-mini")
        ids = [tokenizer.encode(token) for token in [" Yes", " No"]]
        print(ids[0], ids[1])
        ids = [tokenizer.encode(token) for token in ["Yes", "No"]]
        print(ids[0], ids[1])



    def refresh_topic_groups_reference_file_dropdown(self):
        """
        Refresh the topic group reference YAML files.
        """
        logging.info("Refreshing the topic group reference YAML files...")
        files = [f for f in os.listdir(REFERENCE_DIR) if os.path.isfile(os.path.join(REFERENCE_DIR, f))]
        files.sort()
        return gr.update(choices=files)

    def load_topic_groups_reference(self, file_name):
        """
        Load topic group reference YAML file.
        """
        logging.info("Loading topic group reference YAML file...")
        try:
            with open(os.path.join(REFERENCE_DIR, file_name), "r", encoding="utf-8") as f:
                return f.read(), gr.update(value=file_name)
        except FileNotFoundError:
            return "", None

    def save_topic_groups_reference(self, content, file_name):
        """
        Save topic group reference YAML file.
        """
        logging.info("Saving topic group reference YAML file...")
        with open(os.path.join(REFERENCE_DIR, file_name), "w", encoding="utf-8") as f:
            f.write(content)
        gr.Info("Topic group reference saved successfully!")

    def safe_text(self, x):
        if x is None:
            return ""
        if isinstance(x, (dict, list)):
            return json.dumps(x, ensure_ascii=False, sort_keys=False)
        return str(x)

    def get_embedding(self, text: str, model: str):
        text = text.replace("\n", " ")
        embedding = self.model_client.embeddings.create(input=[text], model=model).data[0].embedding
        return np.array(embedding, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    
    def create_topic_groups(self, topic_group_reference, embedding_model):
        """
        Create topic groups according to the topic group reference YAML file.
        """
        logging.info("Creating topic groups according to the topic group reference YAML file...")
        reference = yaml.safe_load(topic_group_reference)

        self.topic_group_list = []
        for entry in reference.get("reference", []):
            name = entry.get("topic_group_name")
            description = entry.get("description")
            if not name or not description:
                print(f"Skipping invalid topic group entry: {entry}")
                continue
            topic_group = {
                "name": name,
                "description": description,
                "embedding": self.get_embedding(self.safe_text(description), embedding_model)
            }
            self.topic_group_list.append(topic_group) 
        
        gr.Info("Topic groups created successfully!")
        return gr.update(variant="primary")

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
                            
        # Cross-encoder testing with OpenAI GPT
        prompt = '''
            You are an Assistant responsible for helping detect whether the retrieved metadata is relevant to the topic_group. For a given metadata, you need to output a single token: "Yes" or "No" indicating the retrieved metadata is relevant to the topic_group. A metadata can be relevant to multiple topic_groups. The topic_group is one of the followings:
            (People) Focuses on human development, including health, education, social protection, and addressing issues in fragile and conflict-affected contexts. Subgroups include Education; Gender; Health; Nutrition & Population; Social Protection.
            (Prosperity) Centers on economic growth, jobs, financial inclusion, and effective macroeconomic and fiscal management to boost shared prosperity. Subgroups include Economic Policy; Finance; Institutions; Poverty; Trader, Investment and Competitiveness.
            (Planet) Addresses environmental challenges, climate change adaptation and mitigation, sustainable food systems, water and sanitation, and a livable planet. Subgroups include Agriculture and Food; Climate Change; Environment; Social Sustainability; Water.
            (Infrastructure) Encompasses the development of reliable and sustainable energy, digital connectivity, and urban infrastructure to improve lives and connect communities. Subgroups include Energy & Extractives; Global Infrastructure Finance; Transport; Urban, Resilience and Land.
            (Digital) Focuses on leveraging digital technologies for development, connecting communities, and enhancing digital services. Subgroups include Connectivity; Data Infrastructure; Cybersecurity; Digital Industry and Jobs; Digital Services.
                        
            Metadata: {{"name": "Fertilizer consumption (% of fertilizer production)", "definition_long": "Fertilizer consumption measures the quantity of plant nutrients used per unit of arable land. Fertilizer products cover nitrogenous, potash, and phosphate fertilizers (including ground rock phosphate). Traditional nutrients--animal and plant manures--are not included. For the purpose of data dissemination, FAO has adopted the concept of a calendar year (January to December). Some countries compile fertilizer data on a calendar year basis, while others are on a split-year basis."}}
            Topic Group: Addresses environmental challenges, climate change adaptation and mitigation, sustainable food systems, water and sanitation, and a livable planet. Subgroups include Agriculture and Food; Climate Change; Environment; Social Sustainability; Water.
            Relevant: Yes

            Metadata: {{"name": "Literacy rate, youth female (% of females ages 15-24)", "definition_long": "Youth literacy rate is the percentage of people ages 15-24 who can both read and write with understanding a short simple statement about their everyday life."}}
            Topic Group: Encompasses the development of reliable and sustainable energy, digital connectivity, and urban infrastructure to improve lives and connect communities. Subgroups include Energy & Extractives; Global Infrastructure Finance; Transport; Urban, Resilience and Land.
            Relevant: No          

            Metadata: {{"name": "Income share held by second 20%", "definition_long": "Percentage share of income or consumption is the share that accrues to subgroups of population indicated by deciles or quintiles. Percentage shares by quintile may not sum to 100 because of rounding."}}
            Topic Group: Centers on economic growth, jobs, financial inclusion, and effective macroeconomic and fiscal management to boost shared prosperity. Subgroups include Economic Policy; Finance; Institutions; Poverty; Trader, Investment and Competitiveness.
            Relevant: Yes    

            Metadata: {{"name": "Percentage of individuals using the internet (ITU), "definition_long": "Proportion of individuals who used the Internet from any location in the last three months. Access can be via a fixed or mobile network. The indicator can be disaggregated: i) by location of user (Rural/Urban); ii) by age of user (14 years and younger, 15-24 years, 25-74 years, 75 years and older); iii) by gender of user (Male/Female)."}}
            Topic Group: Focuses on human development, including health, education, social protection, and addressing issues in fragile and conflict-affected contexts. Subgroups include Education; Gender; Health; Nutrition & Population; Social Protection.
            Relevant: No  
            
            Metadata: {metadata}
            Topic Group: {topic_group}
            Relevant:
        '''

        response = self.model_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an Assistant responsible for deciding if metadata is relevant to the given topic group. Reply only 'Yes' or 'No'."},
                {"role": "user", "content": prompt.format(metadata=metadata, topic_group=topic_group)}
            ],
            temperature=0,
            logprobs=True,
            logit_bias={11377: 1, 3004: 1},
        )

        print(response.choices[0])
        print(response.choices[0].message.content)
        print(response.choices[0].logprobs)

        return (
            response.choices[0].message.content,
            response.choices[0].logprobs.content[0].logprob,
        )

    
    def calculate_semantic_similarities(self, project_title, embedding_model):
        metadata_all = self.fetch_me_project_metadata(project_title)
        metadata_selected = {
            "name": metadata_all.get("series_description", {}).get("name", ""),
            "definition_long": metadata_all.get("series_description", {}).get("definition_long", ""),
            # "methodology": metadata_all.get("series_description", {}).get("methodology", ""),
            # "dimensions": metadata_all.get("series_description", {}).get("dimensions", "")
        }        
        metadata_embedding = self.get_embedding(self.safe_text(metadata_selected), embedding_model)
        baseline_text = "Education; Gender; Health; Nutrition & Population; Social Protection; Economic Policy; Finance; Institutions; Poverty; Trader, Investment and Competitiveness; Agriculture and Food; Climate Change; Environment; Social Sustainability; Water; Energy & Extractives; Global Infrastructure Finance; Transport; Urban, Resilience and Land; Data Infrastructure; Cybersecurity; Digital Industry and Jobs; Digital Services"
        baseline_embedding = self.get_embedding(baseline_text, embedding_model)
        baseline_similarity_score = self.cosine_similarity(metadata_embedding, baseline_embedding)

        topic_group_names = []
        topic_group_descriptions = []
        cosine_similarity_scores = []
        cross_encoding_scores = []
        for topic_group in self.topic_group_list:
            topic_group_names.append(str(topic_group.get("name", "")).strip() or "unnamed")
            topic_group_descriptions.append(str(topic_group.get("description", "")).strip() or "none")
            topic_group_embedding = topic_group.get("embedding", [])
            cosine_similarity = self.cosine_similarity(metadata_embedding, topic_group_embedding) 
            cosine_similarity_scores.append(cosine_similarity)
            cross_encoding_prediction, cross_encoding_logprob = self.get_cross_encoding(self.safe_text(metadata_selected), str(topic_group.get("description", "")).strip())
            cross_encoding_probability  = math.exp(cross_encoding_logprob)
            cross_encoding_score = cross_encoding_probability * -1 + 1 if cross_encoding_prediction == "No" else cross_encoding_probability
            cross_encoding_scores.append(cross_encoding_score)

        print(cross_encoding_scores)
        theta = topic_group_names + [topic_group_names[0]] if topic_group_names else []
        r_cosine_similarity_scores = cosine_similarity_scores + [cosine_similarity_scores[0]] if cosine_similarity_scores else []
        r_cosine_similarity_scores = np.array(r_cosine_similarity_scores, dtype=float)
        r_cross_encoding_scores = np.array(cross_encoding_scores, dtype=float)

        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=r_cosine_similarity_scores, theta=theta, fill="toself", name="cosine similarity", subplot="polar1"))
        fig.add_trace(go.Scatterpolar(r=r_cross_encoding_scores, theta=theta, fill="toself", name="cross-encoding score", subplot="polar2"))
        fig.update_layout(
            title="Semantic Similarity (metadata vs topics)",
            showlegend=True,
            margin=dict(l=20, r=20, t=50, b=20),
            polar1=dict(
                radialaxis=dict(
                    visible=True,
                    range=[baseline_similarity_score/2, 0.4]
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
    
    
    def handler(self):
        """
        Set up the Gradio Blocks UI for interacting with the search system.
        
        Returns:
            gr.Blocks: The Gradio interface definition.
        """
        with gr.Blocks().queue(max_size=50) as handler:

            # UI section for agent description
            gr.Markdown(
                """
                ### The Error-Scanner is a cross-field agent built to detect evident errors, inconsistencies, and encoding issues throughout the metadata.
                """
            )

            # UI section for selecting GPT model
            guide_md1 = gr.Markdown(
                """
                ### 1️⃣ Select an embedding model.
                """
            )
            embedding_model_radio = gr.Radio(["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"], value="text-embedding-3-small", label="Embedding models")

            # UI section for defining topic groups
            guide_md2 = gr.Markdown(
                """
                ### 2️⃣ Define your topic groups in YAML format.
                """
            )
            with gr.Row():
                reference_file_dropdown = gr.Dropdown(choices=None, container=False, interactive=True)
                load_reference_file_btn = gr.Button("Load File")
            topic_groups_reference_textbox = gr.Textbox(value=None, lines=30, label="Define/Edit Topic Groups")
            with gr.Row():
                new_reference_file_name_textbox = gr.Textbox(value=None, container=False, max_lines=1, interactive=True)
                save_topic_groups_reference_button = gr.Button("Save Fils As")
            create_topic_groups_button = gr.Button("Create topic groups", variant="primary")
            
            # UI section for selecting an indicator
            guide_md3 = gr.Markdown(
                """
                ### 3️⃣ Select a project from the Metadata Editor          
                """
            )
            me_collection_dropbox = gr.Dropdown(choices=[], label="Select a collection", value=None, interactive=True)
            me_project_dropbox = gr.Dropdown(choices=[], label="Select a project", value=None, interactive=True)


            # UI section for calculating semantic distances
            guide_md4 = gr.Markdown(
                """
                ### 4️⃣ Calculate semantic distances to topic groups           
                """
            )
            calculate_semantic_similarities_button = gr.Button("Calculate Semantic Distances", variant="primary")
            topic_group_radar_plot = gr.Plot(label="Topic Group Radar Chart")

            # Actions
            load_reference_file_btn.click(fn=self.load_topic_groups_reference, inputs=[reference_file_dropdown], outputs=[topic_groups_reference_textbox, new_reference_file_name_textbox])
            save_topic_groups_reference_button.click(
                fn=self.save_topic_groups_reference, inputs=[topic_groups_reference_textbox, new_reference_file_name_textbox]
            ).then(
                fn=self.refresh_topic_groups_reference_file_dropdown,
                inputs=None,
                outputs=[reference_file_dropdown],
            )
            create_topic_groups_button.click(fn=self.create_topic_groups, inputs=[topic_groups_reference_textbox, embedding_model_radio], outputs=[create_topic_groups_button])
            handler.load(self.fetch_me_collection_list, inputs=None, outputs=[me_collection_dropbox])
            handler.load(self.refresh_topic_groups_reference_file_dropdown, inputs=None, outputs=[reference_file_dropdown])
            me_collection_dropbox.change(fn=self.fetch_me_project_list, inputs=[me_collection_dropbox], outputs=[me_project_dropbox])
            calculate_semantic_similarities_button.click(fn=self.calculate_semantic_similarities, inputs=[me_project_dropbox, embedding_model_radio], outputs=[topic_group_radar_plot])
        
        return handler