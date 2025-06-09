# 🚀 AI Creative Pipeline

Transform text prompts into stunning 3D models through an intelligent AI-powered pipeline.

![Pipeline Flow](https://i.imgur.com/JQhG3xS.png)

## 🌟 Features

* **Local LLM Processing**: Uses DeepSeek/LLaMA for creative prompt expansion
* **Multi-Stage Generation**: Text → Image → 3D model conversion
* **Persistent Memory**: Remembers all your creations for future reference
* **One-Click Operation**: Simple API endpoint for end-to-end generation
* **Production Ready**: Docker support and comprehensive logging

## 🛠 Tech Stack

* **Core AI**: DeepSeek/LLaMA (local LLM)
* **Generation**: Openfabric Text-to-Image & Image-to-3D apps
* **Backend**: Python + FastAPI
* **Memory**: SQLite + FAISS for vector similarity
* **Deployment**: Docker container support

## 🚀 Quick Start

### Prerequisites

* Python 3.8+
* NVIDIA GPU (recommended)
* Docker (optional)

### Installation

```bash
git clone https://github.com/yourusername/ai-creative-pipeline.git
cd ai-creative-pipeline
pip install -r requirements.txt
```

### Running Locally

```bash
python main.py
```

Access the API at: [http://localhost:8888/swagger-ui](http://localhost:8888/swagger-ui)

### Docker Deployment

```bash
docker build -t ai-pipeline .
docker run -p 8888:8888 ai-pipeline
```

## 📡 API Endpoints

### POST /execution

Generate 3D models from text prompts

**Request:**

```json
{
  "prompt": "A futuristic city at sunset"
}
```

**Successful Response:**

```json
{
  "status": "success",
  "original_prompt": "A futuristic city at sunset",
  "enhanced_prompt": "A cyberpunk metropolis with neon-lit skyscrapers reflecting the golden sunset...",
  "image_path": "generated_assets/image_20240315_123456.png",
  "model_path": "generated_assets/model_20240315_123456.glb",
  "generation_time": "45.23s",
  "cuda_memory_used": "7843.2 MB"
}
```

## 🧠 Memory System

The pipeline remembers all your creations:

### GET /memory/search?query=cyberpunk

```json
{
  "results": [
    {
      "prompt": "cyberpunk city at night",
      "timestamp": "2024-03-14T18:23:45",
      "assets": {
        "image": "path/to/image.png",
        "model": "path/to/model.glb"
      }
    }
  ]
}
```

## 🏗 Project Structure

```
ai-creative-pipeline/
├── main.py                # FastAPI application entrypoint
├── pipeline.py            # Core generation logic
├── memory/                # Persistent storage system
│   ├── vector_db.py       # FAISS similarity search
│   └── sqlite_db.py       # Asset metadata storage
├── generated_assets/      # All created images and 3D models
├── Dockerfile             # Container configuration
└── requirements.txt       # Python dependencies
```

## 🛠 Customization

### Configuring LLM

Edit `config/llm_config.yaml` to change models:

```yaml
llm:
  model_name: "facebook/opt-1.3b"
  temperature: 0.7
  max_length: 200
```

### Changing Openfabric Apps

Modify `config/app_config.yaml`:

```yaml
openfabric:
  text_to_image: "f0997a01-d6d3-a5fe-53d8-561300318557"
  image_to_3d: "69543f29-4d41-4afc-7f29-3d51591f11eb"
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📜 License

MIT License - see `LICENSE` file for details.
