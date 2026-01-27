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

# Biến toàn cục
models = { "llm": None, "vector": None, "driver": None, "bus_bot": None }

# --- 2. QUẢN LÝ TÀI NGUYÊN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Server Starting...")
    try:
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            models["llm"] = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("✅ Gemini Loaded")
    except: logger.error("❌ Gemini Error")

    try:
        loop = asyncio.get_running_loop()
        models["vector"] = await loop.run_in_executor(None, lambda: SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder"))
        logger.info("✅ Vector Model Loaded")
    except: logger.error("❌ Vector Error")

    try:
        if NEO4J_URI:
            driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), keep_alive=True, max_connection_lifetime=200, max_connection_pool_size=50)
            await driver.verify_connectivity()
            models["driver"] = driver
            logger.info("✅ Neo4j Connected")
    except: logger.error("❌ Neo4j Error")

    try:
        loop = asyncio.get_running_loop()
        models["bus_bot"] = await loop.run_in_executor(None, BusBotV13)
        logger.info("✅ BusBot Loaded")
    except: logger.error("❌ BusBot Error")

    logger.info("🎉 SERVER READY!")
    yield
    if models["driver"]: await models["driver"].close()
    models.clear()

app = FastAPI(title="Travel AI Backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
if not os.path.exists("images_items"): os.makedirs("images_items")
app.mount("/images_items", StaticFiles(directory="images_items"), name="images")

@app.get("/")
async def health_check(): return {"status": "alive"}

# --- 4. HELPER FUNCTIONS ---
def normalize_text(text):
    if not text: return ""
    if isinstance(text, list): text = next((t for t in text if t), "")
    return unicodedata.normalize('NFC', str(text)).lower().strip()

def clean_entity_name(raw_text: str, use_alias: bool = False) -> str:
    """Hàm chuẩn hóa tên địa điểm"""
    if not raw_text: return ""
    cleaned = raw_text.lower().strip()[:200]
    
    # 1. Cắt bỏ cụm từ thừa (Noise Removal)
    # [FIX QUAN TRỌNG]: Đã xóa "và", "cần", "với", "lưu ý" khỏi danh sách cắt bỏ
    # Để khi hỏi "và cần lưu ý gì", nó không bị cắt mất vế sau.
    stop_phrases = [" bằng ", " hỏi ", " cho ", " để ", " lúc ", " tầm ", " trong ", " ad ", " admin ", " mình ", " bạn ", " giúp "]
    for stop in stop_phrases:
        if stop in cleaned:
            cleaned = cleaned.split(stop)[0].strip()

    # 2. Xóa Prefix (Chỉ xóa phần đầu câu hỏi tìm đường)
    prefixes = ["lộ trình đi xe buýt từ", "lộ trình xe buýt đi từ", "lộ trình xe buýt từ", "lộ trình đi từ", "lộ trình từ", "lộ trình xe buýt về", "lộ trình", "hướng dẫn bắt xe buýt từ", "hướng dẫn đi xe buýt từ", "hướng dẫn đón xe buýt từ", "hướng dẫn bắt xe từ", "hướng dẫn đi từ", "hướng dẫn đường đi từ", "hướng dẫn", "cách đi xe buýt từ", "cách bắt xe buýt từ", "cách đi từ", "tìm đường đi từ", "đường đi xe buýt từ", "đường đi từ", "làm sao để đi từ", "làm sao đi từ", "xe buýt đi từ", "xe buýt từ", "tuyến xe từ", "bắt xe từ", "đón xe từ", "đi xe buýt từ", "đi từ", "đến", "tới", "về", "sang", "qua", "tại", "ở", "khu vực", "tìm đường", "chỉ đường", "cho tôi hỏi về", "thông tin về", "giới thiệu về", "biết gì về", "có gì", "muốn đi", "đang ở", "đứng tại", "làm thế nào để đi"]
    prefixes.sort(key=len, reverse=True)
    for word in prefixes:
        if cleaned.startswith(word):
            cleaned = cleaned.replace(word, "", 1).strip()
    
    cleaned = cleaned.replace(" - ", " ").replace("-", " ")
            
    suffixes = [" như thế nào", " thế nào", " ra sao", " làm sao", " ở đâu", " chỗ nào", " nhé", " nha", " nhỉ", " vậy", " ạ", " hả", " đi", " vui", " ngon", " đẹp"]
    for suffix in suffixes:
        if cleaned.endswith(suffix): cleaned = cleaned[:-len(suffix)].strip()

    # 4. Xử lý Alias - DANH SÁCH FULL 100%
    if use_alias or True:
        ENTITY_ALIASES = {
            "dinh độc lập": "hội trường thống nhất", "hội trường thống nhất": "hội trường thống nhất", "nhà thờ đức bà": "vương cung thánh đường chính tòa đức bà sài gòn", "bưu điện thành phố": "bưu điện trung tâm sài gòn", "bến nhà rồng": "bến nhà rồng", "bảo tàng hồ chí minh": "bến nhà rồng", "bảo tàng hcm": "bến nhà rồng", "chợ lớn": "chợ bình tây", "chợ bến thành": "chợ bến thành", "landmart 81": "landmark 81", "landmark 81": "landmark 81", "bitexco": "tòa nhà bitexco financial tower", "lăng bác": "lăng chủ tịch hồ chí minh", "thảo cầm viên": "thảo cầm viên sài gòn", "sở thú": "thảo cầm viên sài gòn", "suối tiên": "khu du lịch văn hóa suối tiên", "đầm sen": "công viên văn hóa đầm sen", "phố đi bộ": "phố đi bộ nguyễn huệ", "bùi viện": "phố đi bộ bùi viện", "hồ con rùa": "công trường quốc tế",
            "sài gòn": "thành phố hồ chí minh", "hcm": "thành phố hồ chí minh", "tphcm": "thành phố hồ chí minh", "tp hcm": "thành phố hồ chí minh", "tp.hcm": "thành phố hồ chí minh",
            "tân sơn nhất": "sân bay tân sơn nhất", "ga quốc nội": "sân bay tân sơn nhất", "ga quốc tế": "sân bay tân sơn nhất", "phi trường": "sân bay tân sơn nhất", "tsn": "sân bay tân sơn nhất", "bx miền đông": "bến xe miền đông", "bxmd": "bến xe miền đông", "bx miền tây": "bến xe miền tây", "bxmt": "bến xe miền tây", "bx an sương": "bến xe an sương", "bx chợ lớn": "bến xe chợ lớn", "bx buýt sài gòn": "trạm điều hành xe buýt sài gòn", "công viên 23/9": "công viên 23 tháng 9", "cv 23/9": "công viên 23 tháng 9",
            "đh quốc gia": "đại học quốc gia thành phố hồ chí minh", "đhqg": "đại học quốc gia thành phố hồ chí minh", "khtn": "đại học khoa học tự nhiên", "đh khoa học tự nhiên": "đại học khoa học tự nhiên", "xhnv": "đại học khoa học xã hội và nhân văn", "nhân văn": "đại học khoa học xã hội và nhân văn", "đh bách khoa": "đại học bách khoa", "bách khoa": "đại học bách khoa", "hcmut": "đại học bách khoa", "đh quốc tế": "đại học quốc tế", "iu": "đại học quốc tế", "cntt": "đại học công nghệ thông tin", "uit": "đại học công nghệ thông tin", "đh tdt": "đại học tôn đức thắng", "tdt": "đại học tôn đức thắng", "tdtu": "đại học tôn đức thắng", "tôn đức thắng": "đại học tôn đức thắng", "đh văn lang": "đại học văn lang", "vlu": "đại học văn lang", "văn lang": "đại học văn lang", "trường văn lang": "đại học văn lang", "hutech": "đại học công nghệ thành phố hồ chí minh", "đh hutech": "đại học công nghệ thành phố hồ chí minh", "đh công nghệ": "đại học công nghệ thành phố hồ chí minh", "ueh": "đại học kinh tế thành phố hồ chí minh", "đh kinh tế": "đại học kinh tế thành phố hồ chí minh", "sư phạm kỹ thuật": "đại học sư phạm kỹ thuật", "spkt": "đại học sư phạm kỹ thuật", "hcmute": "đại học sư phạm kỹ thuật", "fpt": "đại học fpt", "rmit": "đại học rmit", "đh sài gòn": "đại học sài gòn", "sgu": "đại học sài gòn", "đh mở": "đại học mở thành phố hồ chí minh", "ou": "đại học mở thành phố hồ chí minh", "đh công nghiệp": "đại học công nghiệp thành phố hồ chí minh", "iuh": "đại học công nghiệp thành phố hồ chí minh", "đh luật": "đại học luật thành phố hồ chí minh", "hcmul": "đại học luật thành phố hồ chí minh", "đh y dược": "đại học y dược thành phố hồ chí minh", "ump": "đại học y dược thành phố hồ chí minh", "đh giao thông vận tải": "đại học giao thông vận tải", "gtvt": "đại học giao thông vận tải", "đh nông lâm": "đại học nông lâm", "nlu": "đại học nông lâm", "đh sư phạm": "đại học sư phạm thành phố hồ chí minh", "hcmue": "đại học sư phạm thành phố hồ chí minh",
            "bv chợ rẫy": "bệnh viện chợ rẫy", "chợ rẫy": "bệnh viện chợ rẫy", "bv 115": "bệnh viện nhân dân 115", "bệnh viện 115": "bệnh viện nhân dân 115", "bv gia định": "bệnh viện nhân dân gia định", "nhân dân gia định": "bệnh viện nhân dân gia định", "bv ung bướu": "bệnh viện ung bướu", "ung bướu": "bệnh viện ung bướu", "bv nhi đồng": "bệnh viện nhi đồng", "nhi đồng 1": "bệnh viện nhi đồng 1", "nhi đồng 2": "bệnh viện nhi đồng 2", "bv từ dũ": "bệnh viện từ dũ", "từ dũ": "bệnh viện từ dũ", "bv hùng vương": "bệnh viện hùng vương", "hùng vương": "bệnh viện hùng vương", "bv đại học y dược": "bệnh viện đại học y dược", "bv tai mũi họng": "bệnh viện tai mũi họng", "bv mắt": "bệnh viện mắt",
            "nguyễn hữu thọ": "đường nguyễn hữu thọ", "cmt8": "đường cách mạng tháng tám", "cách mạng tháng 8": "đường cách mạng tháng tám", "đbp": "đường điện biên phủ", "điện biên phủ": "đường điện biên phủ", "nvl": "đường nguyễn văn linh", "nguyễn văn linh": "đường nguyễn văn linh", "hbt": "đường hai bà trưng", "hai bà trưng": "đường hai bà trưng", "nkkn": "đường nam kỳ khởi nghĩa", "nam kỳ khởi nghĩa": "đường nam kỳ khởi nghĩa", "pxl": "đường phan xích long", "phan xích long": "đường phan xích long", "ntmk": "đường nguyễn thị minh khai", "nguyễn thị minh khai": "đường nguyễn thị minh khai", "pvđ": "đường phạm văn đồng", "phạm văn đồng": "đường phạm văn đồng", "võ văn kiệt": "đại lộ võ văn kiệt", "ql1a": "quốc lộ 1a", "ql13": "quốc lộ 13", "xa lộ hà nội": "xa lộ hà nội"
        }
        sorted_aliases = sorted(ENTITY_ALIASES.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            if alias in cleaned: cleaned = cleaned.replace(alias, ENTITY_ALIASES[alias])
    
    cleaned = re.sub(r'(?i)\b(.+?)\s+\1\b', r'\1', cleaned)
    cleaned = re.sub(r'(?i)\b(.+?)\s+\1\b', r'\1', cleaned)
    cleaned = cleaned.replace("-", " ")
    cleaned = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned)
    cleaned = cleaned.replace("đại học đại học", "đại học").replace("trường trường", "trường").replace("thành phố thành phố", "thành phố").replace("đường đường", "đường")
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.title()

async def lookup_bus_stop(name: str):
    driver = models["driver"]
    if not driver: return name
    clean_name = normalize_text(name)
    async with driver.session() as session:
        q1 = "MATCH (b:BusStop) WHERE toLower(b.name) = $name RETURN b.name LIMIT 1"
        r1 = await session.run(q1, name=clean_name)
        rec1 = await r1.single()
        if rec1: return rec1["b.name"]
        q2 = "MATCH (b:BusStop) WHERE toLower(b.name) CONTAINS $name RETURN b.name LIMIT 1"
        r2 = await session.run(q2, name=clean_name)
        rec2 = await r2.single()
        if rec2: return rec2["b.name"]
    return name 

async def hybrid_search_tourism(text: str, province_filter: Union[str, List[str]] = None):
    vec_model = models["vector"]
    driver = models["driver"]
    if not vec_model or not driver: return [] 
    
    # Quan trọng: Clean nhẹ để giữ keyword cho RAG
    text = clean_entity_name(text)
    search_text_norm = normalize_text(text)
    records = []
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            await driver.verify_connectivity()
            async with driver.session() as session:
                p_filter = normalize_text(province_filter) if province_filter else ""
                if len(p_filter) > 2:
                    cypher_loc = """
                    MATCH (node:Searchable)-[:LOCATED_IN]->(p:Province)
                    WHERE toLower(p.name) CONTAINS $p_filter
                    AND (toLower(node.name) CONTAINS $text OR toLower(node.content) CONTAINS $text)
                    RETURN elementId(node) as id, node, 1.5 as score, p.name as province_name, 'location_filter' as source_type LIMIT 15
                    """
                    r_loc = await session.run(cypher_loc, p_filter=p_filter, text=search_text_norm)
                    records.extend([record.data() async for record in r_loc])

                if len(records) < 5: 
                    loop = asyncio.get_running_loop()
                    vector = await loop.run_in_executor(None, lambda: vec_model.encode(text).tolist())
                    cypher_vector = "CALL db.index.vector.queryNodes('searchable_index', 50, $vector) YIELD node, score OPTIONAL MATCH (node)-[:LOCATED_IN]->(p:Province) RETURN elementId(node) as id, node, score, p.name as province_name, 'vector' as source_type"
                    cypher_keyword = "MATCH (node:Searchable) WHERE toLower(node.name) CONTAINS $text OPTIONAL MATCH (node)-[:LOCATED_IN]->(p:Province) RETURN elementId(node) as id, node, 1.0 as score, p.name as province_name, 'keyword' as source_type LIMIT 20"
                    r1 = await session.run(cypher_vector, vector=vector)
                    records.extend([record.data() async for record in r1])
                    r2 = await session.run(cypher_keyword, text=search_text_norm)
                    records.extend([record.data() async for record in r2])
                break 
        except:
            if attempt == max_retries - 1: return []
            await asyncio.sleep(1)

    unique_results = {}
    for r in records:
        uid = r['id']
        if province_filter:
            p_filters = province_filter if isinstance(province_filter, list) else [province_filter]
            if not any(normalize_text(pf) in normalize_text(r.get('province_name')) for pf in p_filters): r['score'] *= 0.1 
        if r['source_type'] == 'vector' and r['score'] < 0.65: continue
        if uid not in unique_results:
            unique_results[uid] = r
            unique_results[uid]['score'] = 10.0 if r['source_type'] in ['keyword', 'location_filter'] else r['score']
    
    return sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)[:10]

class ChatRequest(BaseModel):
    question: str
    province: Optional[Union[str, List[str]]] = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        llm = models["llm"]
        bus = models["bus_bot"]
        if not llm or not bus or not models["vector"]: return {"answer": "Hệ thống đang khởi động.", "sources": [], "images": []}

        raw_question = request.question.strip()
        if len(raw_question) > 500: return {"answer": "Câu hỏi quá dài.", "sources": [], "images": []}
        if not raw_question: return {"answer": "Bạn chưa nhập câu hỏi.", "sources": [], "images": []}
        question = re.sub(r'[\\/*.,?]+$', '', raw_question).strip()
        lower_q = question.lower()
        logger.info(f"REQ: {question}")

        TOXIC_KEYWORDS = ["ngu", "dốt", "chó", "cút", "biến", "đần", "óc", "điên", "khùng", "vô dụng", "như l", "như c", "đm", "đkm", "dm", "vcl", "đéo", "mày", "tao", "bot lỏ", "rác rưởi", "phế vật", "cc", "cl"]
        toxic_pattern = r'\b(' + '|'.join(map(re.escape, TOXIC_KEYWORDS)) + r')\b'
        if re.search(toxic_pattern, lower_q):
            return {"answer": "Vui lòng sử dụng ngôn ngữ lịch sự. Tôi là trợ lý ảo và luôn sẵn sàng hỗ trợ bạn một cách tôn trọng.", "sources": [], "images": []}

        OFF_TOPIC_KEYWORDS = ["code", "lập trình", "python", "java", "sql", "database", "bug", "lỗi", "error", "máy tính", "laptop", "pc", "màn hình xanh", "blue screen", "cpu", "ram", "fps", "giật lag", "cài đặt", "crack", "hack", "virus", "reset", "windows", "linux", "system.exit", "thuốc", "bệnh", "đau", "khám", "bác sĩ", "ung thư", "tiểu đường", "mang thai", "chứng khoán", "bitcoin", "tiền ảo", "vay", "lãi suất", "xổ số", "lô đề", "cá độ", "chính trị", "đảng", "nhà nước", "phản động", "giết", "súng", "bom", "khủng bố", "tự tử", "sex", "làm tình", "18+", "khiêu dâm", "gái", "trai", "tán tỉnh", "yêu đương", "giải toán", "phương trình", "đạo hàm", "tích phân", "vật lý", "hóa học"]
        off_topic_pattern = r'\b(' + '|'.join(map(re.escape, OFF_TOPIC_KEYWORDS)) + r')\b'
        if re.search(off_topic_pattern, lower_q):
            is_valid_context = any(x in lower_q for x in ["ở đâu", "địa chỉ", "đường nào", "xe buýt", "đi ntn", "tham quan", "du lịch", "mặc gì", "trang phục"])
            if not is_valid_context:
                return {"answer": "Xin lỗi, tôi chỉ hỗ trợ **Du lịch & Giao thông Bus TP.HCM**. Tôi không giải đáp các câu hỏi ngoài phạm vi này.", "sources": [], "images": []}

        intent = "off_topic" 
        location_filter = None
        try:
            router_prompt = f"""
            VAI TRÒ: Router Phân loại Intent & Xác thực Địa lý.
            NHIỆM VỤ:
            1. Phân tích địa điểm.
            2. KIỂM TRA ĐỊA LÝ:
               - Hỏi xe buýt NGOÀI TP.HCM -> "out_of_scope_transport".
               - Hỏi Du lịch NGOÀI TP.HCM (Ví dụ: Núi Bà Đen, Đà Lạt) -> "tourism_vn" (ĐỂ RAG TRẢ LỜI).
               - TRONG TP.HCM -> "bus_hcm" hoặc "tourism_vn".
            CÁC NHÓM: "bus_hcm", "tourism_vn", "greeting", "out_of_scope_transport", "off_topic".
            CÂU HỎI: "{question}"
            JSON: {{ "intent": "...", "location": "..." }}
            """
            res = await asyncio.wait_for(asyncio.to_thread(llm.generate_content, router_prompt), timeout=4.0)
            clean_res = res.text.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(clean_res)
            intent = parsed.get("intent", "off_topic")
            location_filter = parsed.get("location")
        except:
            if any(w in lower_q for w in ["chào", "hello"]): intent = "greeting"
            elif any(w in lower_q for w in ["xe buýt", "bus", "tuyến", "trạm", "đi từ", "đến", "cách đi"]): intent = "bus_hcm"
            else: intent = "off_topic" 

        if location_filter: location_filter = clean_entity_name(str(location_filter))
        logger.info(f"🔍 INTENT: {intent} | Loc: {location_filter}")

        if intent == "off_topic": return {"answer": "Xin lỗi, tôi chỉ hỗ trợ **Du lịch & Giao thông Bus TP.HCM**.", "sources": [], "images": []}
        if intent == "out_of_scope_transport": return {"answer": "Hệ thống chưa cập nhật dữ liệu về phương tiện/khu vực này (Chỉ hỗ trợ Bus TP.HCM).", "sources": [], "images": []}
        if intent == "greeting":
            if any(x in lower_q for x in ["cảm ơn", "thank"]): return {"answer": "Dạ không có chi ạ! 🥰", "sources": [], "images": []}
            return {"answer": "Chào bạn! Tôi là trợ lý du lịch & giao thông AI. Bạn cần tìm đường xe buýt (TP.HCM) hay thông tin du lịch?", "sources": [], "images": []}

        if intent in ["bus_hcm", "tourism_vn"]:
            final_response = {"answer": "", "sources": [], "images": [], "path_coords": []}
            start_loc, end_loc = None, None
            is_bus_query = False
            
            match = re.search(r'(?:từ|ở)\s+(.+?)\s+(?:đến|tới|về|sang|qua|ra)\s+(.*)', question, re.IGNORECASE)
            if match:
                is_bus_query = True
                start_loc = match.group(1).strip()
                end_loc = match.group(2).strip()
                if " và " in end_loc.lower(): end_loc = re.split(r' và ', end_loc, flags=re.IGNORECASE)[0].strip()
            elif intent == "bus_hcm":
                seps = [" đến ", " tới ", " về ", " sang ", " qua ", " ra "]
                found_sep = next((s for s in seps if s in lower_q), None)
                if found_sep:
                    idx = lower_q.find(found_sep)
                    if idx != -1:
                        is_bus_query = True
                        start_raw = question[:idx].strip()
                        end_loc = question[idx+len(found_sep):].strip()
                        prefixes = ["lộ trình đi xe buýt từ", "đi từ", "từ", "tìm đường từ", "làm sao để đi xe buýt", "làm thế nào để đi"]
                        lower_start = start_raw.lower()
                        for p in prefixes:
                            if lower_start.startswith(p):
                                start_raw = start_raw[len(p):].strip()
                                break
                        start_loc = start_raw

            bus_info_text = ""
            
            if is_bus_query and start_loc and end_loc:
                try:
                    loop = asyncio.get_running_loop()
                    # L0
                    logger.info(f"Bus L0 (Raw): {start_loc} -> {end_loc}")
                    bus_result = await loop.run_in_executor(None, lambda: bus.solve_route(start_loc, end_loc))

                    # Locking Logic: Chỉ sửa thằng nào bị lỗi
                    if bus_result.get("status") == "error" or "không tìm thấy" in str(bus_result.get("message", "")).lower():
                        error_msg = str(bus_result.get("message", "")).lower()
                        is_start_ok = "điểm đến" in error_msg
                        is_end_ok = "điểm đi" in error_msg
                        
                        db_start = start_loc
                        db_end = end_loc
                        
                        if not is_start_ok: db_start = await lookup_bus_stop(start_loc)
                        if not is_end_ok: db_end = await lookup_bus_stop(end_loc)
                        
                        if db_start != start_loc or db_end != end_loc:
                            bus_result = await loop.run_in_executor(None, lambda: bus.solve_route(db_start, db_end))

                    if bus_result.get("status") == "error":
                        error_msg_2 = str(bus_result.get("message", "")).lower()
                        is_start_ok = "điểm đến" in error_msg_2
                        is_end_ok = "điểm đi" in error_msg_2
                        
                        clean_start = start_loc if is_start_ok else clean_entity_name(start_loc, use_alias=True)
                        clean_end = end_loc if is_end_ok else clean_entity_name(end_loc, use_alias=True)
                        bus_result = await loop.run_in_executor(None, lambda: bus.solve_route(clean_start, clean_end))

                    if bus_result.get("status") == "error":
                        error_msg_3 = str(bus_result.get("message", "")).lower()
                        is_start_ok = "điểm đến" in error_msg_3
                        is_end_ok = "điểm đi" in error_msg_3
                        
                        prompt = f"""Trích xuất tên địa điểm CHÍNH: "{start_loc}", "{end_loc}". JSON: {{"start_clean": "...", "end_clean": "..."}}"""
                        norm_res = await asyncio.wait_for(asyncio.to_thread(llm.generate_content, prompt), timeout=4.0)
                        norm_data = json.loads(norm_res.text.strip().replace("```json", "").replace("```", ""))
                        
                        ai_start = start_loc if is_start_ok else norm_data.get("start_clean")
                        ai_end = end_loc if is_end_ok else norm_data.get("end_clean")
                        
                        if ai_start and ai_end:
                            bus_result = await loop.run_in_executor(None, lambda: bus.solve_route(ai_start, ai_end))

                    if bus_result.get("status") == "ambiguous":
                        return {"answer": bus_result["message"], "options": bus_result["options"], "context_type": "bus_ambiguity", "original_request": {"start": start_loc, "end": end_loc, "type": bus_result["point_type"]}, "sources": [], "images": []}
                    
                    if bus_result.get("status") == "success":
                        bus_info_text = bus_result['text']
                        final_response["path_coords"] = bus_result.get("path_coords", [])
                        if final_response["path_coords"]:
                            s, e = final_response["path_coords"][0], final_response["path_coords"][-1]
                            bus_info_text += f"\n\n🔗 **[Mở Google Maps]**(https://www.google.com/maps/dir/?api=1&origin={s[1]},{s[0]}&destination={e[1]},{e[0]}&travelmode=transit)"
                    else:
                        bus_info_text = f"Không tìm thấy lộ trình xe buýt phù hợp từ {start_loc} đến {end_loc}."

                except Exception:
                    bus_info_text = "Lỗi khi tìm đường xe buýt."
            
            # --- FIX RAG: Sử dụng TOÀN BỘ CÂU HỎI để giữ ngữ cảnh (trang phục, lưu ý) ---
            # Không dùng end_loc đè lên vì sẽ mất keywords quan trọng
            search_query = clean_entity_name(question)
                
            final_province = request.province if request.province else location_filter
            if final_province and isinstance(final_province, list): search_query = final_province[0]
            
            search_results = await hybrid_search_tourism(search_query, final_province)
            imgs = list(dict.fromkeys([item['node'].get('image_url') for item in search_results if item['node'].get('image_url')]))[:2] if search_results else []
            ctx = "\n".join([f"- {i['node']['name']}: {i['node']['content']}" for i in search_results]) if search_results else "Không có dữ liệu chi tiết."
            
            final_response["sources"] = search_results
            final_response["images"] = imgs

            rag_prompt = f"""
            VAI TRÒ: Chuyên gia Du lịch & Giao thông Việt Nam.
            
            DỮ LIỆU ĐẦU VÀO:
            1. KẾT QUẢ TÌM BUÝT (Nếu có):
            {bus_info_text}
            
            2. THÔNG TIN ĐỊA ĐIỂM/DU LỊCH (RAG):
            {ctx}
            
            CÂU HỎI CỦA NGƯỜI DÙNG: "{question}"
            
            YÊU CẦU TRẢ LỜI:
            - Nếu có thông tin xe buýt, hãy chỉ dẫn lộ trình.
            - QUAN TRỌNG: Nếu người dùng hỏi thêm về trang phục, ăn uống, lịch sử... HÃY ƯU TIÊN DÙNG THÔNG TIN RAG ĐỂ TRẢ LỜI. 
            - Nếu thông tin RAG không có (ví dụ: thiếu quy định trang phục), HÃY SỬ DỤNG KIẾN THỨC CỦA BẠN (Gemini) để đưa ra lời khuyên hợp lý (ví dụ: Dinh Độc Lập cần mặc lịch sự, kín đáo).
            - Nếu hỏi địa điểm ngoài TP.HCM (như Núi Bà Đen, Đà Lạt...), hãy dùng RAG để giới thiệu vị trí và cách đi (nếu biết).
            - Trả lời thân thiện, ngắn gọn, trình bày đẹp bằng Markdown.
            """
            res = await asyncio.to_thread(llm.generate_content, rag_prompt)
            final_response["answer"] = res.text
            
            return final_response

    except Exception as e:
        logger.error(f"Endpoint Error: {e}")
        return JSONResponse(status_code=500, content={"answer": "Hệ thống lỗi.", "sources": [], "images": []})

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