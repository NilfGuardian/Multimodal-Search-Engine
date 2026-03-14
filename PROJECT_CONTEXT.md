# MULTIMODAL SEARCH ENGINE - PROJECT CONTEXT FOR COPILOT

## PROJECT OVERVIEW
I'm building a multimodal search engine as a college project. It allows users to search images using either text queries or another image. The system uses CLIP embeddings and Milvus vector database for similarity search.

## SUBMISSION DEADLINE: Monday (urgent!)

## TECH STACK
- Python 3.9+
- Milvus (vector database) - using Milvus Lite for simplicity
- CLIP model (via transformers library) for embeddings
- FastAPI for backend API
- Streamlit for frontend UI
- PyTorch (backend for CLIP)

## PROJECT STRUCTURE
multimodal-search/
├── images/ # Stored images
├── temp/ # Temporary uploads
├── src/
│ ├── embedding.py # CLIP embedding functions
│ ├── milvus_client.py # Milvus database operations
│ ├── api.py # FastAPI backend
│ └── ui.py # Streamlit frontend
├── tests/ # Test files
├── requirements.txt # Dependencies
├── README.md # Documentation
└── PROJECT_CONTEXT.md # This file

text

## CORE REQUIREMENTS

### 1. Embedding Module (embedding.py)
- Load CLIP model (ViT-B/32) from HuggingFace
- Function: `embed_image(image_path)` -> returns 512-dim vector
- Function: `embed_text(text)` -> returns 512-dim vector
- Handle both file paths and PIL images
- Add error handling for corrupt images
- Cache model to avoid reloading

### 2. Milvus Client (milvus_client.py)
- Use Milvus Lite (no Docker needed)
- Collection schema: id (auto), image_path (string), embedding (float vector)
- Create HNSW index for fast search
- Functions:
  - `insert_image(image_path, embedding)`
  - `search(query_embedding, top_k=5)` -> returns paths and scores
  - `count_images()` -> for debugging
- Auto-create collection if not exists

### 3. FastAPI Backend (api.py)
- Endpoints:
  - POST `/index` - upload and index single image
  - POST `/index/batch` - upload multiple images
  - POST `/search/text?q=query` - text search
  - POST `/search/image` - image search (upload file)
  - GET `/stats` - return indexed image count
- CORS enabled for frontend
- Return JSON with image paths and similarity scores
- Save uploaded files to ./images/ or ./temp/

### 4. Streamlit UI (ui.py)
- Two tabs: "Search" and "Add Images"
- Search tab:
  - Radio button: Text or Image search
  - Text input for text queries
  - File uploader for image search
  - Display results in a 3-column grid
  - Show similarity scores under each image
- Add Images tab:
  - Multiple file uploader
  - Progress bar during indexing
  - Success/failure messages
  - Show total indexed count

### 5. Dataset Requirements
- Start with 50 test images (use any public domain images)
- Support common formats: .jpg, .jpeg, .png
- Handle image resizing (max 224x224 for CLIP)
- Extract and store original filenames

## IMPLEMENTATION NOTES

### Embedding Details
- CLIP input size: 224x224
- Normalize embeddings before storing (unit vectors)
- Use cosine similarity for search
- Batch processing for multiple images

### Milvus Configuration
- Metric type: COSINE
- Index type: HNSW (efConstruction=200, M=16)
- Search params: ef=64
- Collection name: "image_search"

### Error Handling Requirements
- Try-catch for all database operations
- Validate image files before processing
- Return meaningful error messages
- Log errors to console

### Performance Optimizations
- Cache CLIP model in memory
- Batch image processing where possible
- Use async where beneficial
- Lazy loading for UI images

## TESTING REQUIREMENTS
- Test with at least 3 different queries
- Verify both search types work
- Test error cases (invalid files, empty queries)
- Measure response time (< 2 seconds acceptable)

## SUBMISSION PACKAGE NEEDS
1. Working code with all features above
2. README with setup instructions
3. Requirements.txt with all dependencies
4. Short demo video (screen recording)
5. Project report (2-3 pages)

## DEVELOPMENT SEQUENCE (FOLLOW THIS ORDER)
1. Day 1 (Friday): embedding.py + milvus_client.py
2. Day 2 (Saturday): api.py + basic testing
3. Day 3 (Sunday): ui.py + integration + polish
4. Day 4 (Monday): Documentation + video

## CODE QUALITY EXPECTATIONS
- Add docstrings to all functions
- Include type hints
- Comment complex logic
- Follow PEP 8
- No hardcoded paths (use config variables)

## COMMON PITFALLS TO AVOID
- Don't forget to normalize embeddings
- Milvus collection must be loaded before search
- Handle different image modes (RGB vs RGBA)
- Close file handles properly
- Don't block the UI thread

## CURRENT STATUS
[Update this section as you progress]
- [x] Day 1 tasks started
- [x] Day 1 tasks complete
- [x] Day 2 tasks started
- [x] Day 2 tasks complete
- [x] Day 3 tasks started
- [x] Day 3 tasks complete
- [x] Day 4 tasks started
- [x] Day 4 tasks complete
🎯 How to Use This with Copilot
Step 1: Initial Setup
Create your project folder

Create PROJECT_CONTEXT.md and paste the entire prompt above

Open VS Code with this file visible (keeps Copilot in context)

Step 2: For Each File You Create
When you create a new file, start with a comment that references the context:

For embedding.py:

python
# Following the multimodal search project in PROJECT_CONTEXT.md
# Implementing the embedding module with CLIP

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
# [Let Copilot autocomplete the rest]
For milvus_client.py:

python
# Based on PROJECT_CONTEXT.md - Milvus client implementation
# Using Milvus Lite for vector storage

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
# [Let Copilot generate the class structure]
Step 3: Specific Task Prompts for Copilot
When you need help with a specific function, use these prompts in comments:

python
# Copilot, create a function to load CLIP model with caching
# Should return model and processor, use GPU if available

# Copilot, implement batch image embedding function
# Input: list of image paths, Output: list of embeddings
# Handle errors gracefully and skip corrupt images

# Copilot, write Milvus search function with proper index params
# Use HNSW with cosine similarity, return top_k results with scores
Step 4: Debugging with Copilot
When you hit errors, use:

python
# This code is giving [paste error]. Copilot, help fix it:
[your code]

# Copilot, add error handling for when Milvus connection fails
# Should retry once and give meaningful error message
🚀 Quick Start Commands for Copilot
Here are the first prompts to use when you start each file:

In terminal (setup):

bash
# Copilot, generate requirements.txt for this project
# Include all necessary packages with compatible versions
In embedding.py:

python
# Copilot, implement complete MultimodalEmbedder class
# With image and text embedding functions
# Use CLIP model, cache it, handle both file paths and PIL images
# Add normalization to make vectors unit length
In milvus_client.py:

python
# Copilot, create MilvusSearchClient class
# Use Milvus Lite (from milvus import default_server)
# Auto-create collection with proper schema
# Implement insert and search methods
# Add index creation and loading
In api.py:

python
# Copilot, build FastAPI app with all endpoints from PROJECT_CONTEXT.md
# Use the embedder and milvus client we created
# Add CORS middleware
# Include file upload handling
In ui.py:

python
# Copilot, create Streamlit UI with two tabs
# First tab: search (text or image)
# Second tab: upload images
# Call the FastAPI backend
# Display results in grid
💡 Pro Tips for Working with Copilot
Be specific in comments: The more detailed your comment, the better Copilot's suggestion

Accept suggestions incrementally: Don't accept huge code blocks without reviewing

Use the context file: Keep PROJECT_CONTEXT.md open in a tab - Copilot can see it

Iterate quickly: If first suggestion isn't right, add more details and try again

Test as you go: Don't wait until the end to test each function

Example of good prompt engineering:

python
# BAD: "Create search function"
# GOOD: "Create search function that takes query embedding, searches Milvus with cosine similarity, 
#        returns top 5 results with image paths and scores, handles empty results gracefully"
📦 Final Submission Checklist for Copilot
Use this checklist and let Copilot help you generate each item:

python
# Copilot, help me create a comprehensive README.md with:
# - Project title and description
# - Setup instructions (step by step)
# - How to run the application
# - Features demonstrated
# - Screenshots placeholder
# - Technologies used
# - Future improvements

# Copilot, generate a requirements.txt with all dependencies
# Include version numbers that work together

# Copilot, create a demo script that tests all endpoints
# Should verify text search, image search, and indexing work
You're all set! With this master prompt, Copilot effectively becomes your AI development partner, understanding the full context of what you're building. Start with the setup commands, create each file in order, and let Copilot generate the code while you focus on testing and integration. Good luck with your Monday submission!