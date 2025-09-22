import re
import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Set up logging format and level for debugging and monitoring
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


class MethodologyChecker:
    def __init__(self, api_key):
        """
        Initialize the MethodologyChecker object.
        """
        logging.info("Initializing MethodologyChecker object...")


    def handler(self):
        """
        Set up the Gradio Blocks UI for interacting with the search system.
        
        Returns:
            gr.Blocks: The Gradio interface definition.
        """
        with gr.Blocks().queue(max_size=50) as handler:

            # UI section for 
            gr.Markdown(
                """
                ### Under construction...
                """
            )

            # UI section for selecting catalog
            guide_md1 = gr.Markdown(
                """
                ### 1️⃣ Select the data catalog where you want to run MethodologyChecker
                """
            )
            catalog = gr.Radio(["Data360", "Databank", "MicrodataLib", "DDH"], value="Data360", label="Data catalog")

            
        return handler