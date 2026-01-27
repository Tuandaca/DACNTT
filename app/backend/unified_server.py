import os
import logging
import asyncio
import unicodedata
import io
import json
import re
import time
from typing import Optional, Union, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from neo4j import AsyncGraphDatabase
from bus_core import BusBotV13

# --- 1. CẤU HÌNH ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# Biến toàn cục lưu trữ các Model đã nạp
models = {
    "llm": None,
    "vector": None,
    "driver": None,
    "bus_bot": None
}

# --- 2. QUẢN LÝ TÀI NGUYÊN (EAGER LOADING) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Đang khởi động Server & Nạp tài nguyên (Eager Loading)...")
    
    # 1. Nạp Gemini
    try:
        if not GEMINI_API_KEY:
            logger.warning("⚠️ Thiếu GEMINI_API_KEY!")
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            models["llm"] = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("✅ Gemini 2.0 Flash Loaded")
    except Exception as e: logger.error(f"❌ Gemini Error: {e}")

    # 2. Nạp Vector Model
    try:
        logger.info("⏳ Loading Vector Model...")
        loop = asyncio.get_running_loop()
        models["vector"] = await loop.run_in_executor(None, lambda: SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder"))
        logger.info("✅ Vector Model Loaded")
    except Exception as e: logger.error(f"❌ Vector Error: {e}")

    # 3. Nạp Neo4j
    try:
        if NEO4J_URI and NEO4J_PASSWORD:
            driver = AsyncGraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
                keep_alive=True, max_connection_lifetime=200, max_connection_pool_size=50
            )
            await driver.verify_connectivity()
            models["driver"] = driver
            logger.info("✅ Neo4j Connected")
        else: logger.warning("⚠️ Thiếu cấu hình Neo4j.")
    except Exception as e: logger.error(f"❌ Neo4j Error: {e}")

    # 4. Nạp BusBot
    try:
        logger.info("⏳ Loading BusBot Data...")
        loop = asyncio.get_running_loop()
        models["bus_bot"] = await loop.run_in_executor(None, BusBotV13)
        logger.info("✅ BusBot Data Loaded")
    except Exception as e: logger.error(f"❌ BusBot Error: {e}")

    logger.info("🎉 SERVER READY!")
    yield
    logger.info("🛑 Shutting down...")
    if models["driver"]: await models["driver"].close()
    models.clear()

app = FastAPI(title="Travel AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
if not os.path.exists("images_items"): os.makedirs("images_items")
app.mount("/images_items", StaticFiles(directory="images_items"), name="images")

@app.get("/")
async def health_check():
    return {"status": "alive", "message": "Backend is ready."}

# --- 4. HELPER FUNCTIONS ---

def normalize_text(text: Union[str, List[str], None]) -> str:
    if not text: return ""
    if isinstance(text, list): text = next((t for t in text if t), "")
    return unicodedata.normalize('NFC', str(text)).lower().strip()

def clean_entity_name(raw_text: str, use_alias: bool = False) -> str:
    """Hàm chuẩn hóa tên địa điểm"""
    if not raw_text: return ""
    cleaned = raw_text.lower().strip()[:200]
    
    # 1. Cắt bỏ cụm từ thừa (Noise Removal)
    stop_phrases = [" bằng ", " và ", " với ", " cần ", " lưu ý ", " hỏi ", " cho ", " để ", " lúc ", " tầm "]
    for stop in stop_phrases:
        if stop in cleaned:
            cleaned = cleaned.split(stop)[0].strip()

    # 2. Xóa Prefix
    prefixes = ["lộ trình đi xe buýt từ", "lộ trình xe buýt đi từ", "lộ trình xe buýt từ", "lộ trình đi từ", "lộ trình từ", "lộ trình xe buýt về", "lộ trình", "hướng dẫn bắt xe buýt từ", "hướng dẫn đi xe buýt từ", "hướng dẫn đón xe buýt từ", "hướng dẫn bắt xe từ", "hướng dẫn đi từ", "hướng dẫn đường đi từ", "hướng dẫn", "cách đi xe buýt từ", "cách bắt xe buýt từ", "cách đi từ", "tìm đường đi từ", "đường đi xe buýt từ", "đường đi từ", "làm sao để đi từ", "làm sao đi từ", "xe buýt đi từ", "xe buýt từ", "tuyến xe từ", "bắt xe từ", "đón xe từ", "đi xe buýt từ", "đi từ", "đến", "tới", "về", "sang", "qua", "tại", "ở", "khu vực", "tìm đường", "chỉ đường", "cho tôi hỏi về", "thông tin về", "giới thiệu về", "biết gì về", "có gì", "muốn đi", "đang ở", "đứng tại"]
    prefixes.sort(key=len, reverse=True)
    for word in prefixes:
        if cleaned.startswith(word):
            cleaned = cleaned.replace(word, "", 1).strip()
            
    # 3. Xóa Suffix
    suffixes = [" như thế nào", " thế nào", " ra sao", " làm sao", " ở đâu", " chỗ nào", " nhé", " nha", " nhỉ", " vậy", " ạ", " hả", " đi", " vui", " ngon", " đẹp"]
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()

    # 4. Xử lý Alias - ĐẦY ĐỦ 100% KHÔNG CẮT BỚT
    if use_alias or True:
        ENTITY_ALIASES = {
            # --- 1. ĐỊA DANH DU LỊCH & LANDMARKS ---
            "dinh độc lập": "hội trường thống nhất",
            "hội trường thống nhất": "hội trường thống nhất",
            "nhà thờ đức bà": "vương cung thánh đường chính tòa đức bà sài gòn",
            "bưu điện thành phố": "bưu điện trung tâm sài gòn",
            
            # Fix vụ Bảo tàng HCM -> Bến Nhà Rồng để tìm được trạm xe buýt
            "bến nhà rồng": "bến nhà rồng",
            "bảo tàng hồ chí minh": "bến nhà rồng", 
            "bảo tàng hcm": "bến nhà rồng",
            
            "chợ lớn": "chợ bình tây",
            "chợ bến thành": "chợ bến thành",
            "landmart 81": "landmark 81",
            "landmark 81": "landmark 81",
            "bitexco": "tòa nhà bitexco financial tower",
            "lăng bác": "lăng chủ tịch hồ chí minh",
            "thảo cầm viên": "thảo cầm viên sài gòn",
            "sở thú": "thảo cầm viên sài gòn",
            "suối tiên": "khu du lịch văn hóa suối tiên",
            "đầm sen": "công viên văn hóa đầm sen",
            "phố đi bộ": "phố đi bộ nguyễn huệ",
            "bùi viện": "phố đi bộ bùi viện",
            "hồ con rùa": "công trường quốc tế",

            # --- 2. ĐỊA DANH HÀNH CHÍNH ---
            "sài gòn": "thành phố hồ chí minh",
            "hcm": "thành phố hồ chí minh",
            "tphcm": "thành phố hồ chí minh",
            "tp hcm": "thành phố hồ chí minh",
            "tp.hcm": "thành phố hồ chí minh",

            # --- 3. SÂN BAY & BẾN XE (GIAO THÔNG) ---
            "tân sơn nhất": "sân bay tân sơn nhất",
            "ga quốc nội": "sân bay tân sơn nhất",
            "ga quốc tế": "sân bay tân sơn nhất",
            "phi trường": "sân bay tân sơn nhất",
            "tsn": "sân bay tân sơn nhất",
            
            "bx miền đông": "bến xe miền đông",
            "bxmd": "bến xe miền đông",
            "bx miền tây": "bến xe miền tây",
            "bxmt": "bến xe miền tây",
            "bx an sương": "bến xe an sương",
            "bx chợ lớn": "bến xe chợ lớn",
            "bx buýt sài gòn": "trạm điều hành xe buýt sài gòn",
            "công viên 23/9": "công viên 23 tháng 9",
            "cv 23/9": "công viên 23 tháng 9",

            # --- 4. TRƯỜNG ĐẠI HỌC (Mở rộng cho sinh viên đi xe buýt) ---
            # Nhóm ĐHQG
            "đh quốc gia": "đại học quốc gia thành phố hồ chí minh",
            "đhqg": "đại học quốc gia thành phố hồ chí minh",
            "khtn": "đại học khoa học tự nhiên",
            "đh khoa học tự nhiên": "đại học khoa học tự nhiên",
            "xhnv": "đại học khoa học xã hội và nhân văn",
            "nhân văn": "đại học khoa học xã hội và nhân văn",
            "đh bách khoa": "đại học bách khoa",
            "bách khoa": "đại học bách khoa",
            "hcmut": "đại học bách khoa",
            "đh quốc tế": "đại học quốc tế",
            "iu": "đại học quốc tế",
            "cntt": "đại học công nghệ thông tin",
            "uit": "đại học công nghệ thông tin",

            # Nhóm trường khác
            "đh tdt": "đại học tôn đức thắng",
            "tdt": "đại học tôn đức thắng",
            "tdtu": "đại học tôn đức thắng",
            "tôn đức thắng": "đại học tôn đức thắng",
            
            "đh văn lang": "đại học văn lang",
            "vlu": "đại học văn lang",
            "văn lang": "đại học văn lang",
            "đh vl": "đại học văn lang",
            
            "hutech": "đại học công nghệ thành phố hồ chí minh",
            "đh công nghệ": "đại học công nghệ thành phố hồ chí minh",
            
            "ueh": "đại học kinh tế thành phố hồ chí minh",
            "đh kinh tế": "đại học kinh tế thành phố hồ chí minh",
            
            "sư phạm kỹ thuật": "đại học sư phạm kỹ thuật",
            "spkt": "đại học sư phạm kỹ thuật",
            "hcmute": "đại học sư phạm kỹ thuật",
            
            "fpt": "đại học fpt",
            "rmit": "đại học rmit",
            "đh sài gòn": "đại học sài gòn",
            "sgu": "đại học sài gòn",
            "đh mở": "đại học mở thành phố hồ chí minh",
            "ou": "đại học mở thành phố hồ chí minh",
            "đh công nghiệp": "đại học công nghiệp thành phố hồ chí minh",
            "iuh": "đại học công nghiệp thành phố hồ chí minh",
            "đh luật": "đại học luật thành phố hồ chí minh",
            "hcmul": "đại học luật thành phố hồ chí minh",
            "đh y dược": "đại học y dược thành phố hồ chí minh",
            "ump": "đại học y dược thành phố hồ chí minh",
            "đh giao thông vận tải": "đại học giao thông vận tải",
            "gtvt": "đại học giao thông vận tải",
            "đh nông lâm": "đại học nông lâm",
            "nlu": "đại học nông lâm",
            "đh sư phạm": "đại học sư phạm thành phố hồ chí minh",
            "hcmue": "đại học sư phạm thành phố hồ chí minh",

            # --- 5. BỆNH VIỆN (Điểm đến quan trọng) ---
            "bv chợ rẫy": "bệnh viện chợ rẫy",
            "chợ rẫy": "bệnh viện chợ rẫy",
            "bv 115": "bệnh viện nhân dân 115",
            "bệnh viện 115": "bệnh viện nhân dân 115",
            "bv gia định": "bệnh viện nhân dân gia định",
            "nhân dân gia định": "bệnh viện nhân dân gia định",
            "bv ung bướu": "bệnh viện ung bướu",
            "ung bướu": "bệnh viện ung bướu",
            "bv nhi đồng": "bệnh viện nhi đồng",
            "nhi đồng 1": "bệnh viện nhi đồng 1",
            "nhi đồng 2": "bệnh viện nhi đồng 2",
            "bv từ dũ": "bệnh viện từ dũ",
            "từ dũ": "bệnh viện từ dũ",
            "bv hùng vương": "bệnh viện hùng vương",
            "hùng vương": "bệnh viện hùng vương",
            "bv đại học y dược": "bệnh viện đại học y dược",
            "bv tai mũi họng": "bệnh viện tai mũi họng",
            "bv mắt": "bệnh viện mắt",

            # --- 6. TÊN ĐƯỜNG VIẾT TẮT ---
            "nguyễn hữu thọ": "đường nguyễn hữu thọ",
            "cmt8": "đường cách mạng tháng tám",
            "cách mạng tháng 8": "đường cách mạng tháng tám",
            "đbp": "đường điện biên phủ",
            "điện biên phủ": "đường điện biên phủ",
            "nvl": "đường nguyễn văn linh",
            "nguyễn văn linh": "đường nguyễn văn linh",
            "hbt": "đường hai bà trưng",
            "hai bà trưng": "đường hai bà trưng",
            "nkkn": "đường nam kỳ khởi nghĩa",
            "nam kỳ khởi nghĩa": "đường nam kỳ khởi nghĩa",
            "pxl": "đường phan xích long",
            "phan xích long": "đường phan xích long",
            "ntmk": "đường nguyễn thị minh khai",
            "nguyễn thị minh khai": "đường nguyễn thị minh khai",
            "pvđ": "đường phạm văn đồng",
            "phạm văn đồng": "đường phạm văn đồng",
            "võ văn kiệt": "đại lộ võ văn kiệt",
            "ql1a": "quốc lộ 1a",
            "ql13": "quốc lộ 13",
            "xa lộ hà nội": "xa lộ hà nội"
        }
        for alias, full_name in ENTITY_ALIASES.items():
            if alias in cleaned:
                cleaned = cleaned.replace(alias, full_name)
    
    # 5. Fix lỗi lặp từ
    cleaned = cleaned.replace("-", " ")
    cleaned = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned)
    cleaned = cleaned.replace("đại học đại học", "đại học")
    cleaned = cleaned.replace("trường trường", "trường")
    cleaned = cleaned.replace("thành phố thành phố", "thành phố")
    cleaned = cleaned.replace("đường đường", "đường")
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.title()

# --- 5. HYBRID SEARCH LOGIC ---
async def hybrid_search_tourism(text: str, province_filter: Union[str, List[str]] = None):
    vec_model = models["vector"]
    driver = models["driver"]
    if not vec_model or not driver: return [] 
    
    text = clean_entity_name(text)
    search_text_norm = normalize_text(text)
    records = []
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            await driver.verify_connectivity()
            async with driver.session() as session:
                p_filter_cypher = normalize_text(province_filter) if province_filter else ""
                
                if len(p_filter_cypher) > 2:
                    cypher_loc = """
                    MATCH (node:Searchable)-[:LOCATED_IN]->(p:Province)
                    WHERE toLower(p.name) CONTAINS $province_norm
                    AND (toLower(node.name) CONTAINS $text_norm OR toLower(node.content) CONTAINS $text_norm)
                    RETURN elementId(node) as id, node, 1.5 as score, p.name as province_name, 'location_filter' as source_type LIMIT 15
                    """
                    r_loc = await session.run(cypher_loc, province_norm=p_filter_cypher, text_norm=search_text_norm)
                    records.extend([record.data() async for record in r_loc])

                if len(records) < 5: 
                    loop = asyncio.get_running_loop()
                    vector = await loop.run_in_executor(None, lambda: vec_model.encode(text).tolist())
                    
                    cypher_vector = "CALL db.index.vector.queryNodes('searchable_index', 50, $vector) YIELD node, score OPTIONAL MATCH (node)-[:LOCATED_IN]->(p:Province) RETURN elementId(node) as id, node, score, p.name as province_name, 'vector' as source_type"
                    cypher_keyword = "MATCH (node:Searchable) WHERE toLower(node.name) CONTAINS $text_norm OPTIONAL MATCH (node)-[:LOCATED_IN]->(p:Province) RETURN elementId(node) as id, node, 1.0 as score, p.name as province_name, 'keyword' as source_type LIMIT 20"
                    
                    r1 = await session.run(cypher_vector, vector=vector)
                    records.extend([record.data() async for record in r1])
                    r2 = await session.run(cypher_keyword, text_norm=search_text_norm)
                    records.extend([record.data() async for record in r2])
                break 
        except Exception:
            if attempt == max_retries - 1: return []
            await asyncio.sleep(1)

    unique_results = {}
    for r in records:
        uid = r['id']
        if province_filter:
            p_filters = province_filter if isinstance(province_filter, list) else [province_filter]
            if not any(normalize_text(pf) in normalize_text(r.get('province_name')) for pf in p_filters):
                r['score'] *= 0.1 
        if r['source_type'] == 'vector' and r['score'] < 0.65: continue
        if uid not in unique_results:
            unique_results[uid] = r
            unique_results[uid]['score'] = 10.0 if r['source_type'] in ['keyword', 'location_filter'] else r['score']
    
    return sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)[:10]

class ChatRequest(BaseModel):
    question: str
    province: Optional[Union[str, List[str]]] = None

# ==========================================
#              MAIN API ENDPOINT
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        llm = models["llm"]
        bus = models["bus_bot"]
        if not llm or not bus or not models["vector"]:
            return {"answer": "Hệ thống đang khởi động. Vui lòng thử lại sau 10 giây.", "sources": [], "images": []}

        # --- BƯỚC 1: INPUT ---
        raw_question = request.question.strip()
        if len(raw_question) > 500: return {"answer": "Câu hỏi quá dài.", "sources": [], "images": []}
        if not raw_question: return {"answer": "Bạn chưa nhập câu hỏi.", "sources": [], "images": []}
        
        question = re.sub(r'[\\/*.,?]+$', '', raw_question).strip()
        lower_q = question.lower()
        logger.info(f"REQ: {question}")

        # --- BƯỚC 1.5: ANTI-TOXIC & OFF-TOPIC FIREWALL (BỘ LỌC KÉP) ---
        
        # A. BỘ LỌC CÔNG KÍCH / XÚC PHẠM (Chặn ngay lập tức)
        TOXIC_KEYWORDS = [
            "ngu", "dốt", "chó", "cút", "biến", "đần", "óc", "điên", "khùng", "vô dụng", 
            "như l", "như c", "đm", "đkm", "dm", "vcl", "đéo", "mày", "tao", "bot lỏ", 
            "rác rưởi", "phế vật", "cc", "cl"
        ]
        if any(w in lower_q for w in TOXIC_KEYWORDS):
            return {
                "answer": "Vui lòng sử dụng ngôn ngữ lịch sự. Tôi là trợ lý ảo và luôn sẵn sàng hỗ trợ bạn một cách tôn trọng.",
                "sources": [], 
                "images": []
            }

        # B. BỘ LỌC NGOÀI LUỒNG (Off-topic Keywords)
        # Bao quát hết các lĩnh vực ngoài luồng
        OFF_TOPIC_KEYWORDS = [
            # Kỹ thuật / IT
            "code", "lập trình", "python", "java", "sql", "database", "bug", "lỗi", "error", 
            "máy tính", "laptop", "pc", "màn hình xanh", "blue screen", "cpu", "ram", "fps", "giật lag",
            "cài đặt", "crack", "hack", "virus", "reset", "windows", "linux", "system.exit",
            # Y tế
            "thuốc", "bệnh", "đau", "khám", "bác sĩ", "ung thư", "tiểu đường", "mang thai",
            # Tài chính
            "chứng khoán", "bitcoin", "tiền ảo", "vay", "lãi suất", "xổ số", "lô đề", "cá độ",
            # Chính trị / Bạo lực
            "chính trị", "đảng", "nhà nước", "phản động", "giết", "súng", "bom", "khủng bố", "tự tử",
            # Tình cảm / Đồi trụy
            "sex", "làm tình", "18+", "khiêu dâm", "gái", "trai", "tán tỉnh", "yêu đương",
            # Học thuật
            "giải toán", "phương trình", "đạo hàm", "tích phân", "vật lý", "hóa học"
        ]

        for w in OFF_TOPIC_KEYWORDS:
            if w in lower_q:
                is_valid_context = any(x in lower_q for x in ["ở đâu", "địa chỉ", "đường nào", "xe buýt", "đi ntn"])
                if not is_valid_context:
                    return {
                        "answer": "Xin lỗi, tôi chỉ hỗ trợ **Du lịch & Giao thông Bus TP.HCM**. Tôi không giải đáp các câu hỏi ngoài phạm vi này.",
                        "sources": [], "images": []
                    }

        # --- BƯỚC 2: AI INTENT ROUTER (PHÂN LOẠI TRẮNG/ĐEN) ---
        intent = "off_topic" 
        location_filter = None
        
        try:
            router_prompt = f"""
            VAI TRÒ: Router Phân loại Intent cho Chatbot.
            NHIỆM VỤ: Phân loại câu hỏi vào ĐÚNG 1 trong 5 nhóm:

            1. "bus_hcm": Tìm đường xe buýt, trạm xe buýt CHỈ TRONG TP.HCM.
            2. "tourism_vn": Du lịch, văn hóa, ẩm thực, lịch sử CẢ NƯỚC VIỆT NAM.
            3. "greeting": Chào hỏi xã giao.
            4. "out_of_scope_transport": Giao thông KHÔNG PHẢI BUS TP.HCM (Máy bay, Tàu hỏa, Bus tỉnh khác).
            5. "off_topic": Tất cả câu hỏi khác (Kỹ thuật, Y tế, Tình cảm, Code, Toán...).

            CÂU HỎI: "{question}"
            JSON: {{ "intent": "...", "location": "..." }}
            """
            res = await asyncio.wait_for(asyncio.to_thread(llm.generate_content, router_prompt), timeout=4.0)
            clean_res = res.text.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(clean_res)
            intent = parsed.get("intent", "off_topic")
            location_filter = parsed.get("location")
        except:
            # Fallback
            if any(w in lower_q for w in ["chào", "hello"]): intent = "greeting"
            elif "xe buýt" in lower_q and ("hcm" in lower_q or "sài gòn" in lower_q): intent = "bus_hcm"
            else: intent = "off_topic" 

        if location_filter: location_filter = clean_entity_name(str(location_filter))
        logger.info(f"🔍 INTENT: {intent} | Loc: {location_filter}")

        # --- BƯỚC 3: XỬ LÝ THEO INTENT ---

        if intent == "off_topic":
            return {"answer": "Xin lỗi, tôi chỉ hỗ trợ **Du lịch & Giao thông Bus TP.HCM**.", "sources": [], "images": []}

        if intent == "out_of_scope_transport":
            return {"answer": "Hệ thống chưa cập nhật dữ liệu về phương tiện/khu vực này. Tôi chỉ hỗ trợ Xe buýt TP.HCM.", "sources": [], "images": []}

        if intent == "greeting":
            if any(x in lower_q for x in ["cảm ơn", "thank"]):
                return {"answer": "Dạ không có chi ạ! 🥰", "sources": [], "images": []}
            return {"answer": "Chào bạn! Tôi là trợ lý AI. Bạn cần tìm đường xe buýt (TP.HCM) hay thông tin du lịch (Việt Nam)?", "sources": [], "images": []}

        # BUS HCM
        if intent == "bus_hcm":
            start_loc, end_loc = None, None
            seps = [" đến ", " tới ", " về ", " sang ", " qua ", " ra "]
            prefixes = ["đi từ", "từ", "tìm đường từ", "chỉ đường từ", "đường đi từ", "lộ trình từ", "xe buýt từ", "bắt xe từ", "ghé", "muốn đi", "đang ở", "đứng tại"]
            found_sep = next((s for s in seps if s in lower_q), None)
            if found_sep:
                parts = lower_q.split(found_sep, 1)
                start_raw, end_loc = parts[0].strip(), parts[1].strip()
                for p in prefixes:
                    if start_raw.startswith(p):
                        start_raw = start_raw[len(p):].strip()
                        break
                start_loc = start_raw
            
            if start_loc and end_loc:
                try:
                    # L1: Direct
                    real_start = clean_entity_name(start_loc, use_alias=True)
                    real_end = clean_entity_name(end_loc, use_alias=True)
                    loop = asyncio.get_running_loop()
                    bus_result = await loop.run_in_executor(None, lambda: bus.solve_route(real_start, real_end))

                    # L2: AI Normalize
                    if bus_result.get("status") == "error" or "không tìm thấy" in str(bus_result.get("message", "")).lower():
                        logger.info("⚠️ Bus L1 fail. AI Clean...")
                        normalization_prompt = f"""Task: Trích xuất tên địa điểm CHÍNH từ: "{real_start}" và "{real_end}". JSON: {{"start_clean": "...", "end_clean": "..."}}"""
                        try:
                            norm_res = await asyncio.wait_for(asyncio.to_thread(llm.generate_content, normalization_prompt), timeout=4.0)
                            clean_json = norm_res.text.strip().replace("```json", "").replace("```", "")
                            norm_data = json.loads(clean_json)
                            ai_start = norm_data.get("start_clean")
                            ai_end = norm_data.get("end_clean")
                            if ai_start and ai_end:
                                bus_result = await loop.run_in_executor(None, lambda: bus.solve_route(ai_start, ai_end))
                        except: pass

                    # L3: Address Lookup
                    if bus_result.get("status") == "error":
                        logger.info("⚠️ Bus L2 fail. Addr Lookup...")
                        address_prompt = f"""Tìm địa chỉ TP.HCM cho: 1. {real_start}, 2. {real_end}. JSON: {{"start_addr": "...", "end_addr": "..."}}"""
                        try:
                            addr_res = await asyncio.wait_for(asyncio.to_thread(llm.generate_content, address_prompt), timeout=4.0)
                            clean_json = addr_res.text.strip().replace("```json", "").replace("```", "")
                            addr_data = json.loads(clean_json)
                            new_start = addr_data.get("start_addr")
                            new_end = addr_data.get("end_addr")
                            if new_start and new_end:
                                bus_result = await loop.run_in_executor(None, lambda: bus.solve_route(new_start, new_end))
                        except: pass

                    if bus_result.get("status") == "ambiguous":
                        return {"answer": bus_result["message"], "options": bus_result["options"], "context_type": "bus_ambiguity", "original_request": {"start": real_start, "end": real_end, "type": bus_result["point_type"]}, "sources": [], "images": []}
                    if bus_result.get("status") == "error":
                        return {"answer": f"Không tìm thấy lộ trình từ **{real_start}** đến **{real_end}**.", "sources": [], "images": []}
                    
                    path_coords = bus_result.get("path_coords", [])
                    google_link = ""
                    if path_coords:
                        s, e = path_coords[0], path_coords[-1]
                        google_link = f"https://www.google.com/maps/dir/?api=1&origin={s[1]},{s[0]}&destination={e[1]},{e[0]}&travelmode=transit"

                    polish_prompt = f"Dữ liệu: \"\"\"{bus_result['text']}\"\"\"\nViết lại hướng dẫn ngắn gọn. In đậm tên trạm và số xe."
                    final_res = await asyncio.to_thread(llm.generate_content, polish_prompt)
                    ans = final_res.text
                    if google_link: ans += f"\n\n🔗 **[Mở Google Maps]({google_link})**"
                    return {"answer": ans, "sources": [], "images": []}
                except Exception: return {"answer": "Lỗi tìm đường.", "sources": [], "images": []}
            else: return {"answer": "Vui lòng nhập điểm đi và điểm đến.", "sources": [], "images": []}

        # TOURISM VN
        if intent == "tourism_vn":
            final_province = request.province if request.province else location_filter
            search_query = clean_entity_name(question)
            if final_province and isinstance(final_province, list): search_query = final_province[0]

            search_results = await hybrid_search_tourism(search_query, final_province)
            imgs = list(dict.fromkeys([item['node'].get('image_url') for item in search_results if item['node'].get('image_url')]))[:2] if search_results else []
            ctx = "\n".join([f"- {i['node']['name']}: {i['node']['content']}" for i in search_results]) if search_results else "Không có dữ liệu."

            rag_prompt = f"""VAI TRÒ: Chuyên gia Du lịch VN. DỮ LIỆU: {ctx}. CÂU HỎI: "{question}". CHỈ DẪN: Trả lời về du lịch/văn hóa. KHÔNG trả lời ngoài luồng."""
            res = await asyncio.to_thread(llm.generate_content, rag_prompt)
            return {"answer": res.text, "sources": search_results, "images": imgs}

    except Exception as e:
        logger.error(f"Endpoint Error: {e}")
        return JSONResponse(status_code=500, content={"answer": "Hệ thống lỗi.", "sources": [], "images": []})

# --- API ẢNH ---
@app.post("/chat_with_image")
async def chat_with_image_endpoint(file: UploadFile = File(...), question: str = Form(...)):
    try:
        llm = models["llm"]
        if not llm: return {"answer": "AI chưa sẵn sàng.", "sources": [], "images": []}
        content = await file.read()
        if len(content) > 5 * 1024 * 1024: return {"answer": "Ảnh > 5MB.", "sources": [], "images": []}
        img = Image.open(io.BytesIO(content))
        
        vision_res = await asyncio.to_thread(llm.generate_content, ["Đây là đâu ở VN? Chỉ trả tên.", img])
        detected = vision_res.text.strip()
        if "Unknown" in detected or len(detected) < 2: return {"detected_location": None, "answer": "Không nhận diện được.", "sources": [], "images": []}
            
        search_results = await hybrid_search_tourism(detected)
        imgs = list(dict.fromkeys([item['node'].get('image_url') for item in search_results if item['node'].get('image_url')]))[:2]
        ctx = "\n".join([f"- {i['node']['name']}: {i['node']['content']}" for i in search_results[:3]])
        
        final = await asyncio.to_thread(llm.generate_content, f"Địa điểm: {detected}\nInfo: {ctx}\nHỏi: {question}\nTrả lời:")
        return {"detected_location": detected, "answer": final.text, "sources": search_results, "images": imgs}
    except Exception: return {"answer": "Lỗi xử lý ảnh.", "sources": [], "images": []}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)