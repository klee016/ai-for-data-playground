import sys
import uvicorn
import asyncio
import logging
import gradio as gr
from fastapi import FastAPI
from ai_for_data_playground_app import ai_for_data_playground

# Fix for Windows Asyncio ConnectionResetError
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Suppress connection reset errors
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

def suppress_asyncio_errors(loop):
    """ Suppress [WinError 10054] errors that are not critical """
    def exception_handler(loop, context):
        if "exception" in context and isinstance(context["exception"], ConnectionResetError):
            return  # Ignore ConnectionResetError
        loop.default_exception_handler(context)

    loop.set_exception_handler(exception_handler)

# Apply the fix
loop = asyncio.get_event_loop()
suppress_asyncio_errors(loop)

app = FastAPI()

@app.get('/')
async def root():
    return 'AI for Data Playground at /ai_for_data_playground', 200

app = gr.mount_gradio_app(app, ai_for_data_playground, path='/ai_for_data_playground', root_path='/ai_for_data_playground')