import os

txt_folder = "ncert_data_txt"
chunk_folder = "ncert_chunks"
chunk_size = 500  # approx number of words per chunk

os.makedirs(chunk_folder, exist_ok=True)

for filename in os.listdir(txt_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(txt_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        words = text.split()
        chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
        
        for idx, chunk in enumerate(chunks):
            chunk_filename = f"{filename.replace('.txt','')}_chunk{idx+1}.txt"
            chunk_path = os.path.join(chunk_folder, chunk_filename)
            with open(chunk_path, "w", encoding="utf-8") as cf:
                cf.write(" ".join(chunk))

print("All chapters have been chunked into 'ncert_chunks' folder.")
