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
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat, MagenticOneGroupChat, Swarm

# Set up logging format and level for debugging and monitoring
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

AGENTS_MANIFEST_DIR = "agents_manifest_name_guard"
AGENTS_MANIFEST_EXTENSION = ".yml"

class NameGuard:
    def __init__(self, openai_api_key, me_api_key):
        """
        Initialize the NameGuard object.
        """
        logging.info("Initializing NameGuard object...")
        self.openai_api_key = openai_api_key
        self.me_url = 'https://metadataeditor.worldbank.org/index.php'
        self.me_headers = {"X-API-KEY": me_api_key}
        self.model_client = None
        self.create_model_client("gpt-4.1-mini")
        self.agent_list = []
        self.external_termination = ExternalTermination()
    
    def create_model_client(self, gpt_model):
        """
        Create an OpenAI model client.
        """
        logging.info("Creating an OpenAI model client...")
        self.model_client = OpenAIChatCompletionClient(
            model=gpt_model,
            api_key=self.openai_api_key,
            temperature=0,
            seed=1029,
        )

    def refresh_manifest_file_dropdown(self):
        """
        Refresh the list of agents manifest YAML files.
        """
        logging.info("Refreshing the list of agents manifest YAML files...")
        files = [f for f in os.listdir(AGENTS_MANIFEST_DIR) if os.path.isfile(os.path.join(AGENTS_MANIFEST_DIR, f))]
        files.sort()
        return gr.update(choices=files)

    def load_agents_manifest(self, file_name):
        """
        Load agents manifest YAML file.
        """
        logging.info("Loading agents manifest YAML file...")
        try:
            with open(os.path.join(AGENTS_MANIFEST_DIR, file_name), "r", encoding="utf-8") as f:
                return f.read(), gr.update(value=file_name)
        except FileNotFoundError:
            return "", None

    def save_agents_manifest(self, content, file_name):
        """
        Save agents manifest YAML file.
        """
        logging.info("Saving agents manifest YAML file...")
        with open(os.path.join(AGENTS_MANIFEST_DIR, file_name), "w", encoding="utf-8") as f:
            f.write(content)
        gr.Info("Agents manifest saved successfully!")


    def create_agents(self, agents_manifest):
        """
        Create agents according to the agents manifest YAML file.
        """
        logging.info("Creating agents according to the agents manifest YAML file...")
        manifest = yaml.safe_load(agents_manifest)

        self.agent_list = []
        for entry in manifest.get("agents_manifest", []):
            name = entry.get("name")
            system_message = entry.get("system_message")
            if not name or not system_message:
                print(f"Skipping invalid agent entry: {entry}")
                continue
            agent = AssistantAgent(
                name=name,
                model_client=self.model_client,
                system_message=system_message
            )
            self.agent_list.append(agent) 
        
        gr.Info("Agents created successfully!")

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
            return gr.update(choices=collection_list, value="[3] WDI - Education")
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

        project_title_list = [project['title'] for project in project_list]
        project_title_list.sort()
        default_value = project_title_list[0] if project_title_list else None
        return gr.update(choices=project_title_list, value=default_value)

    async def launch_orchestrator(self, project_title, team_preset):
        # Define a termination condition that stops the task if the critic approves.
        text_termination = TextMentionTermination("APPROVE")
        
        # Create a team with agents
        if team_preset == "RoundRobinGroupChat":
            team = RoundRobinGroupChat(self.agent_list, termination_condition=text_termination|self.external_termination)
        elif team_preset == "SelectorGroupChat":
            team = SelectorGroupChat(self.agent_list, model_client=self.model_client, termination_condition=text_termination|self.external_termination)
        elif team_preset == "MagenticOneGroupChat":
            team = MagenticOneGroupChat(self.agent_list, model_client=self.model_client, termination_condition=text_termination|self.external_termination)
        elif team_preset == "Swarm":
            team = Swarm(self.agent_list, termination_condition=text_termination|self.external_termination)

        await team.reset()
        outputs = []
        async for event in team.run_stream(task=f"Indicator name: {project_title}"):
            source = getattr(event, "source", None)
            content = getattr(event, "content", None)
            if content:
                outputs.append(f"## **---------- {str(source)} ----------**\n\n{str(content)}\n\n")
                yield "".join(outputs)
 
    def stop_orchestrator(self):
        self.external_termination.set()


    
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
                ### The Name-Guard module is a field-specific agent designed to check the overall quality of indicator names. It evaluates how clear and understandable a name is from a human perspective.
                """
            )

            # UI section for selecting GPT model
            guide_md1 = gr.Markdown(
                """
                ### 1️⃣ Select a GPT model.
                """
            )
            gpt_model_radio = gr.Radio(["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"], value="gpt-4.1-mini", label="GPT models")

            # UI section for defining agents manifest
            guide_md2 = gr.Markdown(
                """
                ### 2️⃣ Define your agents in YAML format.
                """
            )
            with gr.Row():
                manifest_file_dropdown = gr.Dropdown(choices=None, container=False, interactive=True)
                load_manifest_file_btn = gr.Button("Load File")
            agents_manifest_textbox = gr.Textbox(value=None, lines=30, label="Add/Edit Agents manifest")
            with gr.Row():
                new_manifest_file_name_textbox = gr.Textbox(value=None, container=False, max_lines=1, interactive=True)
                save_agents_manifest_button = gr.Button("Save Fils As")
            create_agents_button = gr.Button("Create agents", variant="primary")
            
            # UI section for selecting an indicator
            guide_md3 = gr.Markdown(
                """
                ### 3️⃣ Select a project from the Metadata Editor          
                """
            )
            me_collection_dropbox = gr.Dropdown(choices=[], label="Select a collection", value=None, interactive=True)
            me_project_dropbox = gr.Dropdown(choices=[], label="Select a project", value=None, interactive=True)


            # UI section for running a team of agents
            guide_md4 = gr.Markdown(
                """
                ### 4️⃣ Launch Agent Orchestrator           
                """
            )
            team_preset_radio = gr.Radio(["RoundRobinGroupChat", "SelectorGroupChat", "MagenticOneGroupChat", "Swarm"], value="RoundRobinGroupChat", label="Team presets")
            with gr.Row():
                launch_orchestrator_button = gr.Button("Launch Orchestrator", variant="primary")
                stop_orchestrator_button = gr.Button("Stop Orchestrator", variant="stop")
            status = gr.Markdown()
        

            # Actions
            gpt_model_radio.change(fn=self.create_model_client, inputs=[gpt_model_radio])
            load_manifest_file_btn.click(fn=self.load_agents_manifest, inputs=[manifest_file_dropdown], outputs=[agents_manifest_textbox, new_manifest_file_name_textbox])
            save_agents_manifest_button.click(
                fn=self.save_agents_manifest, inputs=[agents_manifest_textbox, new_manifest_file_name_textbox]
            ).then(
                fn=self.refresh_manifest_file_dropdown,
                inputs=None,
                outputs=[manifest_file_dropdown],
            )
            create_agents_button.click(fn=self.create_agents, inputs=[agents_manifest_textbox])
            handler.load(self.fetch_me_collection_list, inputs=None, outputs=[me_collection_dropbox])
            handler.load(self.refresh_manifest_file_dropdown, inputs=None, outputs=[manifest_file_dropdown])
            me_collection_dropbox.change(fn=self.fetch_me_project_list, inputs=[me_collection_dropbox], outputs=[me_project_dropbox])
            launch_orchestrator_button.click(fn=self.launch_orchestrator, inputs=[me_project_dropbox, team_preset_radio], outputs=[status])
            stop_orchestrator_button.click(fn=self.stop_orchestrator, inputs=None, outputs=None)
        
        return handler