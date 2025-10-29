import re
import os
import time
import json
import yaml
import math
import uuid
import httpx
import logging
import requests
import numpy as np
import pandas as pd
import gradio as gr
from threading import Lock
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat, MagenticOneGroupChat, Swarm

# Disable SSL certificate verification
http_client = httpx.AsyncClient(verify=False)

# Set up logging format and level for debugging and monitoring
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logging.getLogger("autogen_core").setLevel(logging.WARNING)

AGENTS_MANIFEST_DIR = "assets_name_guard/agents_manifest"
AGENTS_MANIFEST_DEFAULT_FILE = "default_agents_manifest.yml"
AGENT_NAME_MARKER = "----------"

class NameGuard:
    def __init__(self, openai_api_key, me_api_key):
        """
        Initialize the NameGuard object.
        """
        logging.info("Initializing NameGuard object...")
        self.openai_api_key = openai_api_key
        self.me_url = 'https://metadataeditor.worldbank.org/index.php'
        self.me_headers = {"X-API-KEY": me_api_key}
        # self.gpt_model = "gpt-5-mini"
        # self.openai_model_client = None
        # self.agent_list = []
        # self.external_termination = ExternalTermination()
        self.sessions = {}  # Store data for each user session
        self.lock = Lock()  # Thread-safe access to sessions
    
    def create_session(self):
        """
        Create a unique session ID and initialize a new session dictionary.
        """
        logging.info("Creating a new session...")
        session_id = str(uuid.uuid4())
        with self.lock:
            self.sessions[session_id] = {
                "agent_list": None,
                "external_termination": ExternalTermination(),
                "last_used_time": time.time()
            }
        return session_id

    def get_session(self, session_id=None):
        """
        Retrieve session data for the given session_id.
        """
        with self.lock:
            if session_id not in self.sessions:
                raise ValueError("Session not found or expired.")
            session = self.sessions[session_id]
            session["last_used_time"] = time.time()
            return session

    def delete_session(self, session_id=None):
        """
        Delete a session, freeing its resources.
        """
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

    def session_unloader(self, session_id=None):
        """
        Check if the session has been idle for 60 minutes and delete it.
        """
        session = self.get_session(session_id)
        if (time.time() - session["last_used_time"]) >= 6000:
            self.delete_session(session_id)

    def create_openai_model_client(self, gpt_model):
        """
        Create an OpenAI model client.
        """
        logging.info("Creating an OpenAI model client...")
        temperature = 1 if gpt_model.startswith("gpt-5") else 0
        openai_model_client = OpenAIChatCompletionClient(
            model=gpt_model,
            api_key=self.openai_api_key,
            temperature=temperature,
            seed=1029,
            http_client=http_client
        )
        return openai_model_client

    def refresh_agents_manifest_files_dd(self):
        """
        Refresh the list of agents manifest YAML files.
        """
        logging.info("Refreshing the list of agents manifest YAML files...")
        files = [f for f in os.listdir(AGENTS_MANIFEST_DIR) if os.path.isfile(os.path.join(AGENTS_MANIFEST_DIR, f))]
        files.sort()
        return gr.update(choices=files)

    def load_agents_manifest_file(self, file_name, session_id=None):
        """
        Load agents manifest YAML file.
        """
        logging.info("Loading agents manifest YAML file...")
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

    def create_agents(self, agents_manifest, gpt_model, session_id=None):
        """
        Create agents according to the agents manifest YAML file.
        """
        logging.info("Creating agents according to the agents manifest YAML file...")
        session = self.get_session(session_id)
        manifest = yaml.safe_load(agents_manifest)
        openai_model_client = self.create_openai_model_client(gpt_model)
        agent_list = []
        for entry in manifest.get("agents_manifest", []):
            name = entry.get("name")
            system_message = entry.get("system_message")
            if not name or not system_message:
                print(f"Skipping invalid agent entry: {entry}")
                continue
            agent = AssistantAgent(
                name=name,
                model_client=openai_model_client,
                system_message=system_message
            )
            agent_list.append(agent) 
        session['agent_list'] = agent_list   
        gr.Info("Agents created successfully!")
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

    
    async def start_agents_activity(self, original_name, team_preset, session_id=None):
        session = self.get_session(session_id)
        agent_list = session['agent_list']

        # Define a termination condition that stops the task if the critic approves.
        text_termination = TextMentionTermination("APPROVE")
        
        # Create a team with agents
        if team_preset == "RoundRobinGroupChat":
            team = RoundRobinGroupChat(agent_list, termination_condition=text_termination|session['external_termination'])
        elif team_preset == "SelectorGroupChat":
            team = SelectorGroupChat(agent_list, termination_condition=text_termination|session['external_termination'])
        elif team_preset == "MagenticOneGroupChat":
            team = MagenticOneGroupChat(agent_list, termination_condition=text_termination|session['external_termination'])
        elif team_preset == "Swarm":
            team = Swarm(agent_list, termination_condition=text_termination|session['external_termination'])

        await team.reset()
        outputs = []
        async for event in team.run_stream(task=f"Indicator name: {original_name}"):
            source = getattr(event, "source", None)
            content = getattr(event, "content", None)
            if content:
                outputs.append(f"## **---------- {str(source)} ----------**\n\n{str(content)}\n\n")
                yield "".join(outputs), None

    def extract_json(self, text):
        idx = text.rfind(AGENT_NAME_MARKER)
        text = text[idx:]
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)    
        if match:
            try:
                data = json.loads(match.group(1))
                return json.dumps(data, indent=4, ensure_ascii=False)
            except json.JSONDecodeError:
                return None
        return None
 
    def stop_agents_activity(self, session_id=None):
        session = self.get_session(session_id)
        session['external_termination'].set()


    
    def handler(self):
        """
        Set up the Gradio Blocks UI for interacting with the search system.
        
        Returns:
            gr.Blocks: The Gradio interface definition.
        """
        with gr.Blocks().queue(max_size=500, default_concurrency_limit=200) as handler:

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
            gpt_model_rdo = gr.Radio(["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini"], value="gpt-4.1-mini", label="GPT models")

            # UI section for defining agents
            guide_md2 = gr.Markdown(
                """
                ### 2️⃣ Define your agents in YAML format.
                """
            )
            with gr.Row():
                agents_manifest_files_dd = gr.Dropdown(choices=None, container=False, value=AGENTS_MANIFEST_DEFAULT_FILE, interactive=True, allow_custom_value=True)
                load_agents_manifest_file_btn = gr.Button("Load Agents Manifest", variant="primary")
            agents_manifest_tb = gr.Textbox(value=None, lines=20, max_lines=20, label="Add/Edit agents manifest")
            with gr.Row():
                agents_manifest_file_name_tb = gr.Textbox(value=None, container=False, max_lines=1, interactive=True)
                save_agents_manifest_file_btn = gr.Button("Save File As")
                delete_agents_manifest_file_btn = gr.Button("Delete File")
            create_agents_btn = gr.Button("Create agents", variant="primary")
            
            # UI section for selecting an indicator
            guide_md3 = gr.Markdown(
                """
                ### 3️⃣ Select a project from the Metadata Editor          
                """
            )
            me_collection_dd = gr.Dropdown(choices=[], label="Select a collection", value=None, interactive=True)
            me_project_name_dd = gr.Dropdown(choices=[], label="Select a project", value=None, interactive=True, allow_custom_value=True)


            # UI section for running a team of agents
            guide_md4 = gr.Markdown(
                """
                ### 4️⃣ Launch Agent Orchestrator           
                """
            )
            team_preset_rdo = gr.Radio(["RoundRobinGroupChat", "SelectorGroupChat", "MagenticOneGroupChat", "Swarm"], value="RoundRobinGroupChat", label="Team presets")
            with gr.Row():
                start_agents_activity_btn = gr.Button("Start Agents Activity", variant="primary")
                stop_agents_activity_btn = gr.Button("Stop Agents Activity", variant="stop")
            with gr.Accordion("Agents activity"):                
                agents_activity_md = gr.Markdown(height=200, max_height=200)
            final_output_js = gr.JSON(label="Final output", height=100, max_height=500)
            session_id = gr.Textbox(visible=False)


            # Actions
            handler.load(self.create_session, inputs=None, outputs=[session_id], show_api=True, api_name="name_guard__create_session")
            handler.load(self.fetch_me_collection_list, inputs=None, outputs=[me_collection_dd], show_api=False)
            handler.load(self.refresh_agents_manifest_files_dd, inputs=None, outputs=[agents_manifest_files_dd], show_api=False)
            load_agents_manifest_file_btn.click(
                fn=self.load_agents_manifest_file, 
                inputs=[agents_manifest_files_dd, session_id], 
                outputs=[agents_manifest_tb, agents_manifest_file_name_tb], 
                show_api=True, api_name="name_guard__load_agents_manifest_file"
            )
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
                inputs=[agents_manifest_tb, gpt_model_rdo, session_id], 
                outputs=[agents_manifest_tb], 
                show_api=True, api_name="name_guard__create_agents"
            )
            me_collection_dd.change(fn=self.fetch_me_project_list, inputs=[me_collection_dd], outputs=[me_project_name_dd], show_api=False)
            start_agents_activity_btn.click(
                fn=self.start_agents_activity, 
                inputs=[me_project_name_dd, team_preset_rdo, session_id], 
                outputs=[agents_activity_md, final_output_js], 
                show_api=True, api_name="name_guard__start_agents_activity"
            ).then(
                fn=self.extract_json,
                inputs=[agents_activity_md],
                outputs=[final_output_js],
                show_api=False
            )
            stop_agents_activity_btn.click(fn=self.stop_agents_activity, inputs=[session_id], outputs=None, show_api=False)
            # Periodically unload session if idle
            unload_session_timer = gr.Timer(600)
            unload_session_timer.tick(fn=self.session_unloader, inputs=[session_id], show_api=False)
            # On UI unload, run delete_session to clean up resources
            handler.unload(lambda: self.delete_session(session_id))

        return handler