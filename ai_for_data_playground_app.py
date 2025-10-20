import os
import logging
import gradio as gr
from dotenv import load_dotenv

import module_topic_group_radar
import module_error_scanner
import module_name_guard
import module_definition_editor
import module_methodology_checker
import module_unit_normalizer
# import module_relevance-framer
# import module_limitations-reviewer
# import module_sources-attributor
# import module_keywords-generator
# import module_topics-classifier
# import module_acronym_expander
# import module_style_polisher
# import module_link_checker



# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ME_API_KEY = os.getenv("ME_API_KEY")


###################
# modules
###################
intro = gr.Markdown(f"""
Welcome to the AI for Data Playground!
This space is designed for the AI for Data Team to experiment, prototype, and learn together. Our goal is to create a low-barrier environment where ideas can be tried quickly, tested automatically with built-in APIs, and refined into solutions. that may later scale into production. Use this app to explore, break things, and share findings with the team—it’s meant to be a safe sandbox for creativity and innovation.
""")
name_guard = module_name_guard.NameGuard(OPENAI_API_KEY, ME_API_KEY)
name_guard_tab = name_guard.handler()

definition_editor = module_definition_editor.DefinitionEditor(api_key=OPENAI_API_KEY)
definition_editor_tab = definition_editor.handler()

methodology_checker = module_methodology_checker.MethodologyChecker(api_key=OPENAI_API_KEY)
methodology_checker_tab = methodology_checker.handler()

unit_normalizer = module_unit_normalizer.UnitNormalizer(api_key=OPENAI_API_KEY)
unit_normalizer_tab = unit_normalizer.handler()

error_scanner = module_error_scanner.ErrorScanner(OPENAI_API_KEY, ME_API_KEY)
error_scanner_tab = error_scanner.handler()

topic_group_radar = module_topic_group_radar.TopicGroupRadar(OPENAI_API_KEY, ME_API_KEY)
topic_group_radar_tab = topic_group_radar.handler()


################
# Gradio Blocks
################

with gr.Blocks(title="AI for Data Playground").queue(max_size=50, default_concurrency_limit=50) as ai_for_data_playground:

    # UI
    gr.Markdown("# AI for Data Playground")
    intro.render()

    with gr.Tab("Topic-Group-Radar"):
        topic_group_radar_tab.render()    
    with gr.Tab("Error-Scanner"):
        error_scanner_tab.render()
    with gr.Tab("Name-Guard"):
        name_guard_tab.render()
    with gr.Tab("Definition-Editor"):
        definition_editor_tab.render()
    with gr.Tab("Methodology-Checker"):
        methodology_checker_tab.render()
    with gr.Tab("Unit-Normalizer"):
        unit_normalizer_tab.render()
    
