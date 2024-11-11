# Image Generation API

⚠️ Warning: *This documentation is fully AI-generated.*


This project is an Image Generation API built with FastAPI. It includes NSFW content detection using Azure and Mixture of Agents (MoA) classifiers.

## Requirements

- Python 3.12+
- pip

## Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    Create a `.env` file in the `config` directory and add the necessary environment variables:
    ```env
    MODEL=<image_generation_model_available_on_together_ai> (default: "black-forest-labs/FLUX.1-schnell")
    TOGETHER_API_KEY=<your_together_api_key>
    AZURE_ENDPOINT=<your_azure_endpoint>
    AZURE_KEY=<your_azure_key>
    ```

## Running the Application

To run the application, use the following command:
```sh
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --workers <number_of_workers>
```

## Logging

Logs are written to app.log with the following format:

```sh
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## API Endpoints

### Generate Image with TogetherAI

- **URL:** `/generate-image/together`
- **Method:** `POST`
- **Request Body:**
    ```json
    {
        "prompt": "string",
        "width": 512, 
        "height": 512,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "seed": 42,
        "nsfw_prompt_check": true,
        "nsfw_image_check": true
    }
    ```
- **Parameters:**

| Parameter            | Type      | Description                                                                                                  | Required | Default | Constraints                             |
|----------------------|-----------|--------------------------------------------------------------------------------------------------------------|----------|---------|-----------------------------------------|
| `prompt`             | `string`  | Text describing the desired image.                                                                           | Yes      | None    | N/A                                     |
| `width`              | `integer` | Width of the generated image in pixels.                                                                      | No       | 512     | Maximum: 1440                           |
| `height`             | `integer` | Height of the generated image in pixels.                                                                     | No       | 512     | Maximum: 1440                           |
| `num_inference_steps`| `integer` | The number of inference steps for the image generation, affecting quality and detail.                        | No       | 50      | Positive integer                         |
| `guidance_scale`     | `float`   | Degree of alignment to the prompt, balancing creativity and adherence to the prompt (**now obsolete**).      | No       | 7.5     | Obsolete, typically set to 7.5          |
| `seed`               | `integer` | Seed for image generation randomness. Set as a specific integer for reproducibility or random for variation. | No       | Random  | Integer or `random` for unique outputs. |
| `nsfw_prompt_check`  | `boolean` | Checks for NSFW content within the prompt.                                                                   | No       | true    | true or false                           |
| `nsfw_image_check`   | `boolean` | Checks for NSFW content within the generated image.                                                          | No       | true    | true or false                           |


- **Response:**
    - `200 OK`: Base64 encoded image string
    - `400 Bad Request`: NSFW content detected
    - `500 Internal Server Error`: Could not generate image or classify text

### NSFW Text Detection with MoA Classifier

- **URL:** `/nsfw-text-detection/moa-clf`
- **Method:** `POST`
- **Request Body:**
    ```json
    {
        "prompt": "string"
    }
    ```
- **Response:**
    - `200 OK`: Content assessment result
    - `500 Internal Server Error`: Could not classify text

## Running the Application with Docker
To start the application using Docker, follow these steps:  

1. Build the Docker Image: Navigate to the directory containing the Dockerfile and run the following command to build the Docker image:  <pre>docker build -t your_image_name . </pre>

2. Run the Docker Container:  Use the following command to run the Docker container, mapping port 8000 of the container to port 8000 on the host:  <pre>docker run -p 8000:8000 your_image_name </pre>

3. Access the Application:  Open your web browser and navigate to http://localhost:8000 to access the application.