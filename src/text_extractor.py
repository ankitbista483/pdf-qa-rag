from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 

class ExtractText:
    def __init__(self,file_path,filename= None):
        self.file_path = file_path
        self.filename = filename if filename else os.path.basename(self.file_path)
        self.chunks = []
        self.metadatas = []
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = 450,
            chunk_overlap = 100,
            length_function = len,
            separators=["\n\n","\n","."," ",""]
        )

    def extract_text(self):
        self.reader = PdfReader(self.file_path)
        for page_index,page_text in enumerate(self.reader.pages):
            text = page_text.extract_text()
            if text:
                chunks = self.splitter.split_text(text)
                for chunk in chunks:
                    self.chunks.append(chunk)
                    self.metadatas.append({"Pages":page_index+1,"Source": os.path.basename(self.filename)})

