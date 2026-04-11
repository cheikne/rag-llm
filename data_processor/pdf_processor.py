# import json
# import re
# import fitz  # PyMuPDF
# import requests
# import io
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# class PDFProcessor:
#     def __init__(self, chunk_size=500, chunk_overlap=50):
#         """
#         Initialize the processor with specific chunking parameters.
#         """
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
        
#         # Recursive splitter: prioritize splitting by paragraphs, then sentences, then words.
#         # This keeps semantic context together for better Embedding/BM25 results.
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", ".", " ", ""]
#         )

#     def clean_text(self, text):
#         """
#         Sanitize raw text extracted from PDF to remove noise.
#         """
#         # 1. Join words split by hyphens at the end of a line (e.g., "over-\nfitting" -> "overfitting")
#         text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
#         # 2. Replace single newlines with spaces (PDFs often have hard breaks mid-sentence)
#         text = text.replace('\n', ' ')
        
#         # 3. Remove multiple consecutive whitespaces
#         text = re.sub(r'\s+', ' ', text)
        
#         # 4. Optional: Remove non-printable characters if any
#         text = re.sub(r'[^\x00-\x7F]+', ' ', text) 
        
#         return text.strip()

#     def process_pdf(self, pdf_path, document_id, max_chunks=None):
#         """
#         Extract text from PDF, clean it, chunk it, and return a list of dictionaries.
#         """
#         chunks_list = []
        
#         try:
#             # doc = fitz.open(pdf_path)
#             if pdf_path.startswith("http"):
#                 print(f"Downloading PDF from: {pdf_path}")
#                 response = requests.get(pdf_path)
#                 response.raise_for_status() # Check for errors
#                 # Open the PDF from bytes in memory
#                 doc = fitz.open(stream=io.BytesIO(response.content), filetype="pdf")
#             else:
#                 # Standard local file opening
#                 doc = fitz.open(pdf_path)
#             full_text = ""

#             # Step 1: Iterate through pages and collect raw text
#             for page_num in range(len(doc)):
#                 page = doc.load_page(page_num)
#                 full_text += page.get_text("text") + " "
            
#             # Step 2: Clean the aggregated text
#             cleaned_text = self.clean_text(full_text)

#             # Step 3: Split text into chunks
#             raw_chunks = self.text_splitter.split_text(cleaned_text)

#             # Step 4: Structure into the desired JSON format
#             for i, chunk_content in enumerate(raw_chunks):
#                 # Control: Stop if max_chunks limit is reached
#                 if max_chunks is not None and len(chunks_list) >= max_chunks:
#                     print(f"Stopping: reached max_chunks limit ({max_chunks})")
#                     break
                    
#                 chunk_obj = {
#                     "chunk_id": i + 1,
#                     "document_id": document_id,
#                     "text": chunk_content,
#                     "embedding": [],  # To be filled in Phase 2
#                     "metadata": {
#                         "source": pdf_path,
#                         "character_count": len(chunk_content),
#                         "chunk_index": i
#                     }
#                 }
#                 chunks_list.append(chunk_obj)

#             doc.close()
#             return chunks_list

#         except Exception as e:
#             print(f"Error processing PDF {pdf_path}: {e}")
#             return []

#     def save_to_json(self, data, output_file):
#         """
#         Export the processed chunks to a JSON file.
#         """
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=4, ensure_ascii=False)
#         print(f"Successfully generated: {output_file} with {len(data)} chunks.")

# # --- EXAMPLE USAGE ---
# if __name__ == "__main__":
#     # Initialize with 600 chars size and 10% overlap
#     processor = PDFProcessor(chunk_size=600, chunk_overlap=60)
    
#     # Process PDF and limit output to 20 chunks for testing
#     processed_data = processor.process_pdf(
#         pdf_path="https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/33a6c8e66c62602f9f03ab6a2c632eed_lec8.pdf", 
#         document_id="mit_ml_lecture_08", 
#         max_chunks=20
#     )
    
#     # Save the result
#     if processed_data:
#         processor.save_to_json(processed_data, "rag_database.json")

import json
import re
import io
import requests
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """
        Initialize with chunking parameters and the smart splitter.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def clean_text(self, text):
        """
        Advanced cleaning to remove PDF artifacts and non-semantic noise.
        """
        # 1. Remove non-printable/control characters (like \u0014, \u0015)
        # This keeps only standard text, numbers, and punctuation
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # 2. Normalize Unicode characters (standardize quotes, dashes, etc.)
        import unicodedata
        text = unicodedata.normalize('NFKC', text)

        # 3. Join words split by hyphens due to line breaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # 4. Replace all types of newlines/tabs with a single space
        text = re.sub(r'[\n\r\t]+', ' ', text)
        
        # 5. Remove problematic symbols often found in math PDFs (optional, adjust if needed)
        # This regex removes sequences of dots (...) and strange standalone symbols
        text = re.sub(r'\.{2,}', ' ', text) 
        
        # 6. Final pass: collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def process_pdf(self, pdf_path, document_id, max_chunks=None):
        """
        Extract from URL or Local path, clean and chunk.
        """
        chunks_list = []
        
        try:
            # Handle URL or Local file
            if pdf_path.startswith("http"):
                response = requests.get(pdf_path)
                response.raise_for_status()
                doc = fitz.open(stream=io.BytesIO(response.content), filetype="pdf")
            else:
                doc = fitz.open(pdf_path)

            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + " "
            
            # Deep clean the entire document text
            cleaned_text = self.clean_text(full_text)

            # Create chunks
            raw_chunks = self.text_splitter.split_text(cleaned_text)

            for i, chunk_content in enumerate(raw_chunks):
                # if max_chunks is not None and len(chunks_list) >= max_chunks:
                #     break
                    
                chunks_list.append({
                    "chunk_id": i + 1,
                    "document_id": document_id,
                    "text": chunk_content,
                    "embedding": [],
                    "metadata": {
                        "source": pdf_path,
                        "character_count": len(chunk_content)
                    }
                })

            doc.close()
            return chunks_list

        except Exception as e:
            print(f"Error: {e}")
            return []

    def save_to_json(self, data, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"File saved: {output_file} ({len(data)} chunks)")

# --- TEST ---
if __name__ == "__main__":

    lectures = [
        {"id": "mit_ml_lec1", "url": "https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/d26f49e758fa83b40c8f22496c857f14_lec1.pdf"},
        {"id": "mit_ml_lec08", "url": "https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/33a6c8e66c62602f9f03ab6a2c632eed_lec8.pdf"},
        {"id": "mit_ml_lec21", "url": "https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/c0becfe9e6d659575a8c9e30b90f55dd_lec21.pdf"},
    ]
    print("Starting PDF processing...")

    processor = PDFProcessor(chunk_size=1500, chunk_overlap=150)
    all_chunks = []

    for lec in lectures:
        print(f"Processing {lec['id']}...")
        chunks = processor.process_pdf(lec['url'], lec['id'])
        all_chunks.extend(chunks) # Append them one after another

    processor.save_to_json(all_chunks, "processed_data.json")

    print("All done! Processed data saved to 'processed_data.json'.")
    # url = "https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/33a6c8e66c62602f9f03ab6a2c632eed_lec8.pdf"
    # processor = PDFProcessor(chunk_size=1500, chunk_overlap=150)
    # data = processor.process_pdf(url, "mit_ml_08", max_chunks=5)
    # processor.save_to_json(data, "cleaned_data.json")
