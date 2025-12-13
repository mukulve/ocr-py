import base64
import customtkinter
import easyocr
import cv2
import fitz 
import numpy as np
import ollama
import io
from PIL import Image
from datetime import datetime

def merge_pdfs(pdf_paths):
    result = fitz.open()
    for pdf in pdf_paths:
        with fitz.open(pdf) as mfile:
            result.insert_pdf(mfile)
    result.save(filename=f"{datetime.now().time()}_result.pdf")

def create_annotated_pdf_per_page(img, page_results, output_path):
    pil_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(pil_img)
    pdf_bytes = io.BytesIO()
    pil_img.save(pdf_bytes, format='PDF')
    pdf_bytes.seek(0)

    doc = fitz.open("pdf", pdf_bytes.read())
    page = doc[0]

    for bbox, text, confidence in page_results:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        rect = fitz.Rect(x_min + 2, y_min + 2, x_max - 2, y_max - 2)
        
        page.insert_textbox(
            rect, 
            text, 
            fontsize=min(12, (y_max - y_min) * 0.6), 
            fontname="helv",
            color=(1, 0, 0, 0.7) 
        )

    doc.save(output_path)
    doc.close()


def pdf_to_images(file_path) -> list[np.ndarray]:
    images = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.tobytes(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        images.append(img)
    doc.close()
    return images

def crop_and_encode(img, bbox):
    x_coords = [pt[0] for pt in bbox]
    y_coords = [pt[1] for pt in bbox]
    x_min, y_min = int(min(x_coords)), int(min(y_coords))
    x_max, y_max = int(max(x_coords)), int(max(y_coords))
    
    cropped = img[y_min:y_max, x_min:x_max]
    _, buffer = cv2.imencode('.png', cropped)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64 

def main(files: list[str]):
    images_to_process = []

    for file in files:
        if file.endswith('.pdf'):
            images = pdf_to_images(file)
            for img in images:
                images_to_process.append(img)
        else:
            img = cv2.imread(file)
            images_to_process.append(img)

    reader = easyocr.Reader(['en'], gpu=True)
    counter = 0
    output_pdfs = []
    for img in images_to_process:
        ocr_ollama_results = []
        results = reader.readtext(img)  
        for bbox, text, confidence in results:  
            cropped_b64 = crop_and_encode(img, bbox)

            try:
                cropped_b64 = crop_and_encode(img, bbox)
                ollama_res = ollama.chat(model="blaifa/Nanonets-OCR-s", messages=[{"role": "user", "content": "Extract text from image", "images": [cropped_b64]}])
                corrected_text = ollama_res['message']['content']
                print(f"Ollama corrected: {corrected_text} from {text}")
                ocr_ollama_results.append((bbox, corrected_text, 1.0))
            except Exception as e:
                print(f"Using original text: {text}")
                ocr_ollama_results.append((bbox, text, confidence))

        output_path = f'annotated_output_{counter}.pdf'
        output_pdfs.append(output_path)
        create_annotated_pdf_per_page(img, ocr_ollama_results, output_path)
        counter += 1

    merge_pdfs(output_pdfs)
        



app = customtkinter.CTk()
app.geometry("600x400")
app.title("OCR Processing App")
app.iconbitmap("icon.ico")
app.resizable(False, False)

files_to_process = []

def select_files():
    file = customtkinter.filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf"), ("Image Files", "*.jpg;*.jpeg;*.png")])
    files_to_process.clear()
    files_to_process.append(file)
    text_area.insert("end", f"Selected file: {files_to_process}")

def process():
    if not files_to_process:
        text_area.insert("end", "No file selected!\n")
        return
    main(files_to_process)
    progressbar.set(1)

text_label = customtkinter.CTkLabel(app, text="OCR Processing App")
text_label.pack(padx=20, pady=10)

text_area = customtkinter.CTkTextbox(app, height=100, width=500)
text_area.pack(padx=20, pady=10)

progressbar = customtkinter.CTkProgressBar(app, orientation="horizontal")
progressbar.set(0)
progressbar.pack(padx=20, pady=20)

select_button = customtkinter.CTkButton(app, text="Select File", command=select_files)
select_button.pack(padx=20, pady=10)

button = customtkinter.CTkButton(app, text="Process", command=process)
button.pack(padx=20, pady=20)

app.mainloop()