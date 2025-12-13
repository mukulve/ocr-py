# OCR Tool

## Pipeline
- use easyocr (Python Library) to get all bounding boxes of text
- crop each image to each bounding box area 
- pass cropped text region to Ollama
- Reconstruct a transcribed image 