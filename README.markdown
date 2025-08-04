![Logo](https://github.com/allvoicelab/AllVoiceLab-MCP/raw/main/doc/imgs/logo.jpeg)

<div align="center" style="line-height: 1;">
  <a href="https://github.com/alyrraza/AllVoiceLab-MCP" style="margin: 2px;">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repository-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://www.allvoicelab.com" style="margin: 2px;">
    <img alt="AllVoiceLab" src="https://img.shields.io/badge/Powered_by-AllVoiceLab-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<p align="center">
A voice-cloned RAG (Retrieval-Augmented Generation) chatbot that processes PDF documents and responds in a cloned voice using AllVoiceLab's Model Context Protocol (MCP) for voice cloning and LangChain for retrieval. Upload a PDF, provide an audio sample to clone your voice, and ask questions—get answers in your own voice, offline, without relying on external LLMs.
</p>

## Why Choose This Chatbot?

- **PDF Processing**: Extract text from PDFs and create a vector store for retrieval using LangChain and FAISS.
- **Voice Cloning**: Clone your voice with a 3-15 second audio sample using AllVoiceLab's MCP.
- **RAG Pipeline**: Retrieve relevant document context and generate informative responses with Gemini API.
- **Text-to-Speech (TTS)**: Convert responses to speech in your cloned voice, powered by AllVoiceLab.
- **Streamlit Interface**: User-friendly UI for uploading PDFs, recording/uploading audio, and interacting with the chatbot.
- **Offline Capability**: Voice cloning and TTS run locally after initial setup.

## Quickstart

1. Get your API keys from [Google Cloud](https://cloud.google.com/) for Gemini and [AllVoiceLab](https://www.allvoicelab.com/workbench/api-keys) for voice cloning.
2. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
3. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Set environment variables in a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your-gemini-api-key
   ALLVOICELAB_API_KEY=your-allvoicelab-api-key
   ALLVOICELAB_API_DOMAIN=https://api.allvoicelab.com
   ```
   - Use `https://api.allvoicelab.cn` for Mainland China.
6. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the Streamlit app (usually at `http://localhost:8501`).
2. Upload a PDF to create a vector store for retrieval.
3. Record a 3-15 second audio clip or upload an MP3/WAV file to clone your voice.
4. Click "Submit Audio" to process and clone the voice.
5. Enter a query or transcribe audio input, then click "Submit Query".
6. Listen to responses in your cloned voice (TTS enabled by default).

## Project Structure

- `app.py`: Streamlit application for the chatbot interface.
- `main.py`: Script for testing RAG and voice cloning.
- `config.py`: API keys, embeddings, and system prompt configuration.
- `rag.py`: RAG pipeline using LangChain and Gemini.
- `tts.py`: Voice cloning and TTS using AllVoiceLab's MCP.

## Example

1. Upload a PDF about AI advancements.
2. Record or upload a short audio clip of your voice.
3. Ask: "What are the main AI advancements discussed in the document?"
4. Receive a response in your cloned voice.

## Dependencies

- `streamlit`: Web interface
- `langchain`, `langchain-community`, `langchain-huggingface`: RAG and embeddings
- `faiss-cpu`: Vector storage
- `pydub`, `PyPDF2`, `scipy`: Audio and PDF processing
- `google-generativeai`: Gemini API
- `allvoicelab-mcp`: Voice cloning and TTS

## Troubleshooting

- **Audio Issues**: Ensure FFmpeg is installed and in your system PATH.
- **API Errors**: Verify API keys and domains in `.env`. Check logs at:
  - macOS: `~/.mcp/allvoicelab_mcp.log`
  - Windows: `C:\Users\<Username>\.mcp\allvoicelab_mcp.log`
- **Vector Store Errors**: Use text-based PDFs (not scanned) for proper text extraction.
- Contact [tech@allvoicelab.com](mailto:tech@allvoicelab.com) for API issues.

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
