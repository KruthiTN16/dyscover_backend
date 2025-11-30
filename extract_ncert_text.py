import os
from PyPDF2 import PdfReader

pdf_folder = "ncert_data"
txt_folder = "ncert_data_txt"

os.makedirs(txt_folder, exist_ok=True)

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        txt_filename = filename.replace(".pdf", ".txt")
        txt_path = os.path.join(txt_folder, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

print("All PDFs converted to text files in 'ncert_data_txt' folder.")
