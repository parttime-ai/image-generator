import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.config import AppConfiguration
from app.endpoints import router
from app.nsfw_detection.image_classifier import AzureImageNsfwContentClassifier
from app.nsfw_detection.text_classifier import MixtureOfAgentsClassifier, AzureTextNsfwContentClassifier
from app.together_ai.together_ai import TogetherAI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info("Starting the Image Generation API")
    logger.info("Loading AppSettings")
    app.state.app_configuration = AppConfiguration()

    logger.info("Loading the TogetherAI model")
    app.state.together_ai = TogetherAI(
        model=app.state.app_configuration.model,
        api_key=app.state.app_configuration.together_api_key)

    logger.info("Loading MoA NSFW text detection")
    app.state.moa_clf = MixtureOfAgentsClassifier(api_key=app.state.app_configuration.together_api_key)

    logger.info("Azure Image Content Safety Classifier")
    app.state.azure_image_clf = AzureImageNsfwContentClassifier(
        endpoint=app.state.app_configuration.azure_endpoint,
        key=app.state.app_configuration.azure_key
    )

    logger.info("Azure Text Content Safety Classifier")
    app.state.azure_text_clf = AzureTextNsfwContentClassifier(
        endpoint=app.state.app_configuration.azure_endpoint,
        key=app.state.app_configuration.azure_key
    )

    yield
    # shutdown
    logger.info("Shutting down the Image Generation API")

app = FastAPI(lifespan=lifespan)
app.include_router(router)

logging.basicConfig(level=logging.INFO,
                    filename="app.log",
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


@app.get("/")
async def hello_world(response_class=HTMLResponse):
    logger.info("Hello World endpoint called")
    html_content = """
        <html>
            <head>
                <title>ImageGenerator</title>
                <script>
                    async function callTogetherAI() {
                        const prompt = document.getElementById('together_ai_prompt').value;
                        const response = await fetch('/generate-image/together', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ prompt: prompt })
                        });
                        
                        const errorElement = document.getElementById('together_ai_error');
                        if (!response.ok) {
                            errorElement.textContent = 'Error: ' + response.statusText;
                            return;
                        }
                        const data = await response.json();
                        document.getElementById('together_ai_image').src = 'data:image/png;base64,' + data;
                        errorElement.textContent = '';
                    }
                    
                    async function callMoaClf() {
                        const prompt = document.getElementById('moa_clf_text').value;
                        const response = await fetch('/nsfw-text-detection/moa-clf', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ prompt: prompt })
                        });
                        const data = await response.json();
                        console.log(data);
                        document.getElementById('moa_clf_result').innerHTML = JSON.stringify(data);
                    }
                </script>
            </head>
            <body>
                <h1>Image Generator Demo</h1>
                <div>
                    <input type="text" id="together_ai_prompt" placeholder="TogetherAI Prompt">
                    <button onclick="callTogetherAI()">Call TogetherAI</button>
                    <br>
                    <img id="together_ai_image" src="" alt="TogetherAI Image">
                    <p id="together_ai_error" style="color: red;"></p>
                </div>
                <div>
                    <input type="text" id="moa_clf_text" placeholder="MoaClf">
                    <button onclick="callMoaClf()">Call MoaClf</button>
                    <br>
                    <p id="moa_clf_result"></p>
                </div>
            </body>
        </html>
    """

    return HTMLResponse(html_content, status_code=200)
