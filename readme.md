# OCR Tool

<img width="668" height="496" alt="image" src="https://github.com/user-attachments/assets/20f72ecf-e120-43a3-944d-6c68e9decf41" />


## Pipeline
- use easyocr (Python Library) to get all bounding boxes of text
- crop each image to each bounding box area 
- pass cropped text region to Ollama
- Reconstruct a transcribed image 
