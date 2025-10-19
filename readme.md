AI-Powered Furniture Recommendation & Analytics

A full-stack web application that provides conversational furniture recommendations, AI-generated product descriptions, and an analytics dashboard that includes computer-vision model performance. The project was built as an intern deliverable and combines NLP, CV, and generative AI techniques.

## Highlights

- Natural-language product recommendations using a Retrieval-Augmented Generation (RAG) pipeline
- Semantic search over a vector store (Pinecone) with sentence-transformer embeddings
- AI-generated creative descriptions (Google Gemini)
- Analytics dashboard with EDA and CV model performance (Accuracy, F1, Confusion Matrix)
- Fine-tuned Vision Transformer (ViT) for furniture classification

## Tech stack

- Backend: FastAPI
- Frontend: React, React Router, Recharts
- Vector DB: Pinecone
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Generative AI: Google Gemini (gemini-1.5-flash)
- CV: PyTorch + timm (Vision Transformer)
- Data & Analytics: pandas, scikit-learn

## Repo layout

Root files

- `Data_Analytics.py` — exploratory analysis and visualizations used by the analytics page
- `Model_train.py` — training and evaluation script for the ViT model
- `download_images.py` — helper to prepare/download images for training
- `intern_data_ikarus.csv` — product dataset

Backend

- `Backend/ingest.py` — create embeddings and populate Pinecone index
- `Backend/main.py` — FastAPI app exposing recommendation and analytics endpoints
- `Backend/requirements.txt` — Python dependencies for the backend

Frontend

- `frontend/src/` — React source (App.js, Analytics.js, etc.)
- `frontend/public/` — public assets
- `frontend/package.json` — frontend dependencies and scripts

Images

- `images/`, `images_v3/`, `images_v4/` — labeled image folders used for CV training and experiments

---

## Quickstart (Windows / PowerShell)

These steps get the backend and frontend running locally. They assume you have Conda (or another Python env manager), Python 3.11+, Node.js and npm installed.

1) Clone the repo

```powershell
git clone <your-repo-url>
cd <your-repo-folder>
```

2) Backend: create & activate environment, install deps

```powershell
cd Backend
conda create --name ai_project_env python=3.11 -y
conda activate ai_project_env
pip install -r requirements.txt
```

3) Backend: configuration

Create a `.env` file inside the `Backend/` directory with these variables (example):

```text
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
PINECONE_INDEX_NAME="YOUR_PINECONE_INDEX_NAME"
GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY"
```

Notes:
- Keep real keys secret and don't commit `.env` to source control.
- If you don't have Pinecone or Gemini keys, some functionality will be limited; the server may still serve static analytics pages.

4) (Optional) Populate Pinecone index (one-time)

This will read the dataset, create embeddings, and upload them to your Pinecone index.

```powershell
python ingest.py
```

5) Start the backend server

```powershell
# from Backend/ directory
python -m uvicorn main:app --reload
```

By default the API will be available at: http://127.0.0.1:8000

6) Frontend: install and run

Open a new PowerShell window (keep backend running) and run:

```powershell
cd ..\frontend
npm install
npm start
```

The React app will open at http://localhost:3000 (or the next available port).

---

## Running deliverables

- Run exploratory analytics (prints stats and shows plots):

```powershell
# from project root
conda activate ai_project_env
python Data_Analytics.py
```

- Train/evaluate the CV model:

```powershell
# prepare images (if needed)
python download_images.py
# run training & evaluation
python Model_train.py
```

The training script saves model artifacts and produces an evaluation JSON that the analytics page can read.

## Environment & troubleshooting

- If pip install fails for specific packages, ensure you have a suitable compiler toolchain and correct Python version (3.11 recommended).
- For GPU training with PyTorch, install the matching CUDA-enabled wheel from pytorch.org and update `Backend/requirements.txt` accordingly.
- If the frontend shows CORS errors when calling the backend, confirm the FastAPI server is running and CORS is enabled in `Backend/main.py`.

## Notes for maintainers

- Secrets: add `Backend/.env` to `.gitignore` if not already ignored.
- The project relies on external APIs (Pinecone, Google Gemini); provide mocks or fallbacks for offline development.
- Consider adding small unit tests for endpoint contracts and a CI workflow for linting and tests.

---

If you'd like, I can also:

- Add a minimal `Backend/.env.example` file and update `.gitignore` (low-risk improvement).
- Create short developer scripts in `package.json` or a PowerShell script to streamline local startup.

If you want one of those, tell me which and I'll implement it next.