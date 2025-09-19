import os
import warnings
import pdfplumber
import fitz  # PyMuPDF สำหรับอ่าน PDF
import pytesseract
from PIL import Image, ImageEnhance
import io
import cv2
import numpy as np
from typing import Type, List, Tuple, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict
from chonkie import SemanticChunker
from pythainlp import word_tokenize, sent_tokenize
from pythainlp.util import isthai, reorder_vowels
from dotenv import load_dotenv
import hashlib
import time
import gc
import traceback
import logging
from .qdrant_storage import QdrantStorage, MyEmbedder

# กรอง Warning ที่ไม่จำเป็น (เช่นจาก library ภายนอก)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DocumentSearchTool")

load_dotenv()

# หากจำเป็นให้กำหนด path ของ tesseract (ใน Windows ตัวอย่างเช่น)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class DocumentSearchToolInput(BaseModel):
    """
    สคีมาสำหรับรับข้อมูลอินพุตสำหรับการค้นหาในเอกสาร PDF
    """
    query: str = Field(..., description="คำถามสำหรับค้นหาในเอกสาร")
    context: Optional[str] = Field(None, description="ประวัติการสนทนาก่อนหน้า")
    model_config = ConfigDict(extra="allow")


class DocumentSearchTool(object):
    name: str = "DocumentSearchTool"
    description: str = "ค้นหาข้อมูลเกี่ยวกับ PDPA จากเอกสาร PDF โดยใช้ pythainlp, fitz, pdfplumber และ OCR"
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    model_config = ConfigDict(extra="allow")

    def __init__(self, file_path: str):
        """
        เริ่มต้น DocumentSearchTool ด้วยไฟล์หรือโฟลเดอร์ที่ระบุ
        """
        super().__init__()
        self.file_path = file_path
        self.raw_text = ""
        self.chunks = []
        self.initialized = False
        self.use_vector_db = True  # เปิดใช้งาน vector database
        self.vector_db = None
        # Use file content hash for collection name stability
        if os.path.isdir(file_path):
            # For directories, hash all file contents
            content_hash = hashlib.md5()
            for filename in sorted(os.listdir(file_path)):
                if filename.lower().endswith('.pdf'):
                    with open(os.path.join(file_path, filename), 'rb') as f:
                        while True:
                            chunk = f.read(8192)
                            if not chunk:
                                break
                            content_hash.update(chunk)
            self.file_hash = content_hash.hexdigest()
        else:
            # For single file, hash its content
            with open(file_path, 'rb') as f:
                content = f.read()
            self.file_hash = hashlib.md5(content).hexdigest()
        # เพิ่มตัวแปรสำหรับการจัดการแคชและ garbage collection
        self.image_cache = {}  # เก็บภาพที่สกัดมาแล้วเพื่อใช้ซ้ำ
        self.query_cache = {}  # เก็บผลลัพธ์การค้นหาเพื่อใช้ซ้ำ
        self.last_cache_cleanup = time.time()
        self.cache_ttl = 3600  # ระยะเวลาที่เก็บแคช (วินาที)
        self.last_gc_time = time.time()
        self.gc_interval = 300  # ระยะเวลาระหว่างการทำ garbage collection (วินาที)
        # ตรวจสอบว่าเป็นไฟล์หรือโฟลเดอร์
        #if os.path.isdir(file_path):
        #    self._load_directory()
        #else:
        #    self._load_single_file()

    def _ensure_initialized(self):
        """
        Ensure the tool is initialized before use. This is called lazily on first search.
        """
        if not self.initialized:
            logger.info("Initializing DocumentSearchTool...")
            if os.path.isdir(self.file_path):
                self._load_directory()
            else:
                self._load_single_file()
            #self.initialized = True

    def _load_directory(self):
        """
        โหลดและรวมเนื้อหาจากทุกไฟล์ในโฟลเดอร์
        """
        try:
            self._initialize_vector_db()
            # Check if Qdrant already has data for this collection
            if self.use_vector_db and self.vector_db and self.vector_db.has_data():
                logger.info("Qdrant collection already has data, skipping all extraction and upload.")
                self.initialized = True
                return
            all_text = []
            for filename in os.listdir(self.file_path):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(self.file_path, filename)
                    logger.info(f"Loading file: {filename}")
                    # Force text extraction only for subfiles, avoid premature indexing
                    temp_tool = DocumentSearchTool(file_path)
                    temp_tool.use_vector_db = False
                    temp_tool._ensure_initialized()
                    if temp_tool.raw_text:
                        all_text.append(temp_tool.raw_text)
                        logger.info(f"Successfully loaded {filename} (bytes: {len(temp_tool.raw_text)})")
                    else:
                        logger.warning(f"No text extracted from {filename}")
                    temp_tool.release_resources()
            if all_text:
                self.raw_text = "\n\n".join(all_text)
                self._initialize_tool()
            else:
                logger.error("No valid PDF files found in the directory")
        except Exception as e:
            logger.error(f"Error loading directory: {str(e)}")
            logger.error(traceback.format_exc())

    def _load_single_file(self):
        """
        โหลดเนื้อหาจากไฟล์เดียว
        """
        try:
            if self.vector_db is None:
                self._initialize_vector_db()
            # Check if Qdrant already has data for this collection
            if self.use_vector_db and self.vector_db and self.vector_db.has_data():
                logger.info("Qdrant collection already has data, skipping extraction & indexing; will use existing vectors.")
                self.initialized = True
                return
            self.raw_text = self._extract_text()
            if self.raw_text:
                self._initialize_tool()
        except Exception as e:
            logger.error(f"Error loading single file: {str(e)}")
            logger.error(traceback.format_exc())

    def _initialize_tool(self):
        """
        เริ่มต้นเครื่องมือหลังจากโหลดเนื้อหา
        """
        try:
            if not self.raw_text:
                logger.warning("No text extracted from the document")
                return
            if self.vector_db is None:
                self._initialize_vector_db()
            # Check if data already exists in Qdrant, skip chunk extraction and upload if so
            if self.use_vector_db and self.vector_db and self.vector_db.has_data():
                logger.info("Qdrant collection already has data, skipping chunk extraction and upload.")
                self.initialized = True
                return
            # Only extract chunks if not already in Qdrant
            self.chunks = self._create_chunks(self.raw_text)
            if not self.chunks:
                logger.warning("No chunks created from the text")
                return
            if self.use_vector_db:
                self._index_chunks()
            self.initialized = True
            logger.info(f"DocumentSearchTool initialized successfully with {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error initializing DocumentSearchTool: {str(e)}")
            logger.error(traceback.format_exc())

    def _initialize_vector_db(self):
        """
        Initialize Qdrant vector database
        """
        try:
            self.embedder = MyEmbedder("all-MiniLM-L6-v2")
            self.vector_db = QdrantStorage(
                type=f"doc_{self.file_hash}",
                qdrant_location=os.getenv("QDRANT_URL", "http://localhost:6333"),
                qdrant_api_key=os.getenv("QDRANT_API_KEY", None),
                embedder=self.embedder
            )
            logger.info(f"Qdrant vector database initialized successfully (collection: {self.vector_db.collection_name})")
        except Exception as e:
            logger.error(f"Error initializing Qdrant vector DB: {str(e)}")
            logger.error(traceback.format_exc())
            self.use_vector_db = False

    def _index_chunks(self):
        """
        Index chunks in Qdrant vector database
        """
        try:
            if not self.chunks or not self.vector_db:
                logger.warning("No chunks or vector_db available for indexing")
                return
            for i, chunk in enumerate(self.chunks):
                # Always extract text, even if chunk is a custom object
                if isinstance(chunk, dict) and "text" in chunk:
                    chunk_dict = chunk.copy()
                elif isinstance(chunk, str):
                    chunk_dict = {"text": chunk}
                else:
                    text = getattr(chunk, "text", None)
                    if text is not None:
                        chunk_dict = {"text": text}
                    else:
                        logger.warning(f"Skipping chunk {i}: cannot extract text (type: {type(chunk)})")
                        continue
                chunk_dict["id"] = i  # Use integer index as Qdrant point ID
                self.vector_db.add(chunk_dict)
            logger.info(f"Indexed {len(self.chunks)} chunks in Qdrant vector database (collection: {self.vector_db.collection_name})")
        except Exception as e:
            logger.error(f"Error indexing chunks: {str(e)}")
            logger.error(traceback.format_exc())
            self.use_vector_db = False

    def _is_vector_db_ready(self) -> bool:
        return (
            self.use_vector_db and 
            self.vector_db is not None and 
            self.embedder is not None
        )

    def _process_context(self, context: Optional[str], max_length: int = 1000) -> Optional[str]:
        """
        ประมวลผล context โดยจำกัดขนาดและทำความสะอาด
        """
        if not context:
            return None
            
        # จำกัดขนาด context
        if len(context) > max_length:
            # ตัดเอาเฉพาะส่วนท้ายที่มีขนาด max_length
            context = context[-max_length:]
            # หาตำแหน่งขึ้นบรรทัดใหม่แรกเพื่อไม่ให้ตัดกลางประโยค
            newline_pos = context.find('\n')
            if newline_pos > 0:
                context = context[newline_pos+1:]
                
        # ทำความสะอาด context
        return self._process_thai_text(context)

    def _search_chunks(self, query: str, context: Optional[str] = None) -> List[str]:
        """
        ค้นหาชิ้นส่วนข้อความ (chunks) ที่ตรงกับคำถาม โดยใช้วิธีเปรียบเทียบ token หรือ vector similarity
        """
        try:
            self._cleanup_cache()
            cache_key = self._get_cache_key(query + (context or ""))
            if cache_key in self.query_cache:
                timestamp, result = self.query_cache[cache_key]
                if time.time() - timestamp <= self.cache_ttl:
                    return result
            processed_query = self._process_thai_text(query)
            processed_context = self._process_context(context)
            if self._is_vector_db_ready():
                try:
                    search_query = processed_query
                    if processed_context:
                        search_query = f"Context: {processed_context}\nQuery: {processed_query}"
                    results = self.vector_db.search(search_query, limit=5)
                    chunks = [r["text"] for r in results if "text" in r]
                    self.query_cache[cache_key] = (time.time(), chunks)
                    return chunks
                except Exception as e:
                    logger.error(f"Error using Qdrant vector DB: {str(e)}")
            query_tokens = set(word_tokenize(processed_query))
            results = []
            for chunk in self.chunks:
                # ถ้า chunk เป็น dict ให้ใช้ key "text" หากไม่ให้ถือว่าเป็น string
                chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
                tokens_chunk = set(word_tokenize(chunk_text))
                score = len(query_tokens.intersection(tokens_chunk))
                if score > 0:
                    results.append((score, chunk_text))
            results.sort(key=lambda x: x[0], reverse=True)
            chunks = [text for score, text in results[:10]]
            
            # เก็บผลลัพธ์ในแคช
            self.query_cache[cache_key] = (time.time(), chunks)
            return chunks
        except Exception as e:
            logger.error(f"Error in _search_chunks: {str(e)}")
            return []

    def _run(self, query: str, context: Optional[str] = None) -> str:
        """
        รันการค้นหาข้อมูล:
         - ตรวจสอบว่าคำถามมีคีย์เวิร์ด PDPA หรือไม่
         - ค้นหาและส่งผลลัพธ์เป็นข้อความภาษาไทย
        """
        try:
            self._ensure_initialized()

            # ตรวจสอบว่าเครื่องมือถูกเริ่มต้นสำเร็จหรือไม่
            if not self.initialized:
                return "เครื่องมือค้นหาเอกสารยังไม่พร้อมใช้งาน กรุณาลองใหม่อีกครั้ง"
            
            # ทำความสะอาดแคชที่หมดอายุ
            self._cleanup_cache()
            
            processed_query = self._process_thai_text(query)

            try:
                search_results = self._search_chunks(query, context)
                if search_results:
                    result = "\n____\n".join(search_results)
                else:
                    result = "ไม่พบผลลัพธ์ที่เกี่ยวข้อง"
            except Exception as e:
                logger.error(f"Error in search_chunks: {str(e)}")
                logger.error(traceback.format_exc())
                result = "เกิดข้อผิดพลาดในการค้นหาข้อมูล กรุณาลองใหม่อีกครั้ง"
            
            # ทำ garbage collection เป็นระยะ
            self._perform_gc()
            
            return result
        except Exception as e:
            logger.error(f"Error in _run: {str(e)}")
            logger.error(traceback.format_exc())
            return f"เกิดข้อผิดพลาดในการค้นหาข้อมูล: {str(e)}"
    
    def release_resources(self):
        """
        ปล่อยทรัพยากรที่ใช้ในการประมวลผลเอกสาร
        """
        try:
            # ล้างแคช
            self.image_cache.clear()
            self.query_cache.clear()
            
            # ล้างข้อมูลที่สกัดมา
            self.raw_text = ""
            self.chunks = []
            
            # ลบ collection จาก vector DB ถ้ามี
            if self.vector_db:
                try:
                    self.vector_db.delete()
                except:
                    pass
            
            # ทำ garbage collection
            gc.collect()
            
            logger.info("Resources released successfully")
            return True
        except Exception as e:
            logger.error(f"Error in release_resources: {str(e)}")
            return False

    def _cleanup_cache(self):
        """
        ลบแคชที่หมดอายุเพื่อป้องกันการใช้หน่วยความจำมากเกินไป
        """
        try:
            current_time = time.time()
            if current_time - self.last_cache_cleanup > 300:  # ทำความสะอาดทุก 5 นาที
                expired_keys = []
                for key, (timestamp, _) in self.query_cache.items():
                    if current_time - timestamp > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.query_cache[key]
                
                self.last_cache_cleanup = current_time
        except Exception as e:
            logger.error(f"Error in _cleanup_cache: {str(e)}")

    def _get_cache_key(self, query: str) -> str:
        """
        สร้างคีย์สำหรับแคชจากคำถาม
        """
        return hashlib.md5(query.encode()).hexdigest()

    def _process_thai_text(self, text: str) -> str:
        """
        ประมวลผลข้อความภาษาไทย:
         - ปรับสระลอยให้ถูกต้อง
         - แบ่งประโยคและแยกคำด้วย pythainlp
        """
        try:
            fixed_text = reorder_vowels(text)
            sentences = sent_tokenize(fixed_text)
            processed_sentences = []
            for sentence in sentences:
                words = word_tokenize(sentence)
                processed_sentence = " ".join(words)
                processed_sentences.append(processed_sentence)
            return " ".join(processed_sentences)
        except Exception as e:
            logger.error(f"Error in _process_thai_text: {str(e)}")
            return text  # Return original text if processing fails

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        ปรับปรุงคุณภาพของภาพก่อนทำ OCR เพื่อเพิ่มความแม่นยำ
        """
        # แปลงเป็น grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # ปรับความคมชัด
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # ปรับความสว่าง
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        
        # แปลงเป็น numpy array สำหรับการประมวลผลด้วย OpenCV
        img_array = np.array(image)
        
        # ลด noise
        img_array = cv2.fastNlMeansDenoising(img_array)
        
        # ปรับความคมชัดด้วย adaptive thresholding
        img_array = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # แปลงกลับเป็น PIL Image
        return Image.fromarray(img_array)

    def _extract_embedded_images(self, doc: fitz.Document) -> List[Tuple[int, Image.Image]]:
        """
        สกัดภาพที่ฝังอยู่ใน PDF
        """
        images = []
        for page_num, page in enumerate(doc):
            # ดึงภาพที่ฝังอยู่ในหน้า
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    # ดึงข้อมูลภาพ
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # แปลงเป็น PIL Image
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        # เก็บภาพพร้อมกับหมายเลขหน้า
                        images.append((page_num, pil_image))
                        
                        # เก็บในแคช
                        cache_key = f"{page_num}_{img_index}"
                        self.image_cache[cache_key] = pil_image
                except Exception as e:
                    continue
        
        return images

    def _extract_text(self) -> str:
        """
        สกัดและประมวลผลข้อความจากไฟล์ PDF โดยใช้ pdfplumber และ fitz
        เพิ่มการใช้ OCR ในกรณีที่หน้านั้นไม่ได้มีข้อความ (เช่น สแกนมาเป็นรูป)
        """
        def extract_with_pdfplumber(path: str) -> str:
            text_pp = ""
            try:
                with pdfplumber.open(path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_pp += self._process_thai_text(page_text) + "\n"
                        else:
                            # ใช้ OCR ผ่าน pytesseract โดยใช้ภาพจาก pdfplumber
                            try:
                                # ใช้วิธี to_image() เพื่อแปลงหน้าเป็นภาพ
                                page_image = page.to_image(resolution=300)
                                pil_image = page_image.original  # PIL Image
                                
                                # ปรับปรุงคุณภาพภาพก่อนทำ OCR
                                processed_image = self._preprocess_image(pil_image)
                                
                                # ทำ OCR ด้วยภาษาไทยและอังกฤษ
                                ocr_text = pytesseract.image_to_string(processed_image, lang="tha+eng")
                                if ocr_text.strip():
                                    text_pp += self._process_thai_text(ocr_text) + "\n"
                            except Exception as e:
                                pass
            except Exception as e:
                text_pp += ""
            return text_pp

        def extract_with_fitz(path: str) -> str:
            text_fitz = ""
            try:
                doc = fitz.open(path)
                
                # สกัดภาพที่ฝังอยู่ใน PDF
                embedded_images = self._extract_embedded_images(doc)
                
                for page_num, page in enumerate(doc):
                    page_text = page.get_text().strip()
                    if not page_text:
                        # หากไม่มีข้อความ ให้ใช้ OCR โดย render หน้าด้วย fitz
                        try:
                            # ตรวจสอบว่ามีภาพที่สกัดมาแล้วหรือไม่
                            page_images = [img for p_num, img in embedded_images if p_num == page_num]
                            
                            if page_images:
                                # ใช้ภาพที่สกัดมาแล้ว
                                for img in page_images:
                                    processed_image = self._preprocess_image(img)
                                    ocr_text = pytesseract.image_to_string(processed_image, lang="tha+eng")
                                    if ocr_text.strip():
                                        page_text += self._process_thai_text(ocr_text) + "\n"
                            else:
                                # ถ้าไม่มีภาพที่สกัดมาแล้ว ให้ render หน้าเป็นภาพ
                                pix = page.get_pixmap(dpi=300)
                                image_bytes = pix.tobytes("png")
                                pil_image = Image.open(io.BytesIO(image_bytes))
                                
                                # ปรับปรุงคุณภาพภาพก่อนทำ OCR
                                processed_image = self._preprocess_image(pil_image)
                                
                                ocr_text = pytesseract.image_to_string(processed_image, lang="tha+eng")
                                page_text = ocr_text
                        except Exception as e:
                            page_text = ""
                    if page_text:
                        text_fitz += self._process_thai_text(page_text) + "\n"
                doc.close()
            except Exception as e:
                text_fitz += ""
            return text_fitz

        all_text = ""
        if os.path.isdir(self.file_path):
            for filename in os.listdir(self.file_path):
                if filename.lower().endswith('.pdf'):
                    full_path = os.path.join(self.file_path, filename)
                    text_pp = extract_with_pdfplumber(full_path)
                    text_fitz = extract_with_fitz(full_path)
                    all_text += text_pp + "\n" + text_fitz + "\n"
        else:
            all_text = extract_with_pdfplumber(self.file_path) + "\n" + extract_with_fitz(self.file_path)
        #print(all_text)
        return all_text

    def _create_chunks(self, raw_text: str) -> list:
        """
        สร้าง semantic chunks จากข้อความที่สกัดมาโดยใช้ SemanticChunker
        """
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=128,
            min_sentences=1
        )
        return chunker.chunk(raw_text)

    def _perform_gc(self):
        """
        ทำ garbage collection เป็นระยะเพื่อป้องกันการใช้หน่วยความจำมากเกินไป
        """
        try:
            current_time = time.time()
            if current_time - self.last_gc_time > self.gc_interval:
                gc.collect()
                self.last_gc_time = current_time
        except Exception as e:
            logger.error(f"Error in _perform_gc: {str(e)}")