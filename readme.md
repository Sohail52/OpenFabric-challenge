# 🚀 AI Creative Pipeline

Transform text prompts into stunning 3D models through an intelligent AI-powered pipeline.

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

- Python 3.8+  
- NVIDIA GPU (recommended)  
- Docker (optional)

### Installation

```bash
git clone https://github.com/Sohail52/OpenFabric-challenge.git
cd OpenFabric-challenge
pip install -r requirements.txt
```

### Run Locally

```bash
python main.py
```

Access the API docs at:  
👉 [http://localhost:8888/swagger-ui](http://localhost:8888/swagger-ui)

---

## 🐳 Docker Deployment

```bash
docker build -t ai-pipeline .
docker run -p 8888:8888 ai-pipeline
```

---

## 📡 API Endpoints

### POST `/execution`

Generate 3D models from a simple text prompt.

**Request:**

```json
{
  "prompt": "A futuristic city at sunset"
}
```

**Response:**

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

---

## 🧠 Memory System

Stores all generated assets and metadata for future search and recall.

### GET `/memory/search?query=cyberpunk`

**Response:**

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

---

## 🗂 Project Structure

```
ai-creative-pipeline/
├── main.py                # FastAPI entrypoint
├── pipeline.py            # Core generation logic
├── memory/
│   ├── vector_db.py       # FAISS vector similarity
│   └── sqlite_db.py       # Metadata storage
├── generated_assets/      # Output images & models
├── config/
│   ├── llm_config.yaml    # LLM settings
│   └── app_config.yaml    # Openfabric app IDs
├── Dockerfile             # Docker build config
└── requirements.txt       # Dependencies
```

---

## ⚙️ Customization

### 🔧 Change LLM Config

Edit `config/llm_config.yaml`:

```yaml
llm:
  model_name: "facebook/opt-1.3b"
  temperature: 0.7
  max_length: 200
```

### 🧩 Update OpenFabric Apps

Edit `config/app_config.yaml`:

```yaml
openfabric:
  text_to_image: "f0997a01-d6d3-a5fe-53d8-561300318557"
  image_to_3d: "69543f29-4d41-4afc-7f29-3d51591f11eb"
```

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch:
   ```bash
   git checkout -b feature/awesome-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add awesome feature"
   ```
4. Push and open a pull request

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
