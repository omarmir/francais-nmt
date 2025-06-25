# Offline Neural Machine Translation API

This project provides a FastAPI backend for performing offline neural machine translation (NMT) and interacting with a local Large Language Model (LLM). It includes functionalities for translation, LLM text generation, system monitoring, and glossary management.

## Features

-   **Offline Translation**: Translate English to French using a pre-trained Opus-MT model.
-   **LLM Text Generation**: Interact with the CroissantLLMChat-v0.1 model for text generation based on prompts.
-   **Glossary Management**: Upload, retrieve, replace, and clear a custom CSV glossary to influence LLM responses.
-   **Model Management**: Check model loading status, trigger model reloads, and monitor download progress.
-   **System Monitoring**: Get real-time information on CPU and memory usage.

## Models Used

-   **Translation**: `Helsinki-NLP/opus-mt-en-fr` (and `opus-mt-tc-big-en-fr` is referenced but not actively used in the provided code).
-   **LLM**: `croissantllm/CroissantLLMChat-v0.1`

Models are downloaded and loaded asynchronously on application startup.

## Setup

To set up and run this project, you'll need Python 3.8+ and `venv` for managing virtual environments.

### 1. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
python3 -m venv venv
# On Windows (Command Prompt)
.\venv\Scripts\activate.bat
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# On macOS/Linux
source venv/bin/activate
```
### 2. Install Dependencies
Once your virtual environment is active, install the required Python packages:
`pip install -r requirements.txt`

### 3. Running the Application
The application can be run using uvicorn. Models will be downloaded and loaded on startup. This might take some time depending on your internet connection.
`uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

### 4. API Documentation
The API documentation (Swagger UI) will be available at http://127.0.0.1:8000/docs.

### 5. Glossary Format
The glossary CSV file should contain two columns: `source_term,translation`.

Example `glossary.csv`:
source_term,translation
apple,pomme
banana,banane
orange,orange

## Environment Variables
TRANSLATOR_MODEL_DIR: Specifies the directory where models will be downloaded and stored. Defaults to ./models. You can override this using the TRANSLATOR_MODEL_DIR environment variable.

The glossary is stored in MODEL_DIR/glossary.csv. The maximum number of terms allowed in the glossary is GLOSSARY_MAX_TERMS (currently 500).