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

# --- 2. QUẢN LÝ TÀI NGUYÊN (EAGER LOADING - TẢI TRƯỚC TOÀN BỘ) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Hàm này chạy NGAY KHI SERVER KHỞI ĐỘNG.
    Nó sẽ nạp toàn bộ model vào RAM để sẵn sàng phục vụ.
    """
    logger.info("🚀 Đang khởi động Server & Nạp tài nguyên (Eager Loading)...")
    
    # 1. Nạp Google Gemini
    try:
        if not GEMINI_API_KEY:
            logger.warning("⚠️ Thiếu GEMINI_API_KEY! Chức năng Chat sẽ lỗi.")
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            models["llm"] = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("✅ Gemini 2.0 Flash Loaded")
    except Exception as e:
        logger.error(f"❌ Lỗi nạp Gemini: {e}")

    # 2. Nạp Vector Model (Nặng nhất - Ưu tiên chạy sớm)
    try:
        logger.info("⏳ Đang nạp Vector Model (BKAI Vietnamese)... Vui lòng đợi.")
        # Chạy trong thread riêng để không chặn event loop
        loop = asyncio.get_running_loop()
        models["vector"] = await loop.run_in_executor(None, lambda: SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder"))
        logger.info("✅ Vector Model Loaded")
    except Exception as e:
        logger.error(f"❌ Lỗi nạp Vector Model: {e}")

    # 3. Kết nối Neo4j
    try:
        if NEO4J_URI and NEO4J_PASSWORD:
            driver = AsyncGraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                keep_alive=True,                 # Giữ kết nối sống
                max_connection_lifetime=200,     # Làm mới kết nối sau mỗi 200s
                max_connection_pool_size=50      # Tăng số lượng kết nối đồng thời
            )
            await driver.verify_connectivity()
            models["driver"] = driver
            logger.info("✅ Neo4j Database Connected")
        else:
            logger.warning("⚠️ Thiếu cấu hình Neo4j (URI/PASSWORD).")
    except Exception as e:
        logger.error(f"❌ Lỗi kết nối Neo4j: {e}")

    # 4. Nạp dữ liệu BusBot
    try:
        logger.info("⏳ Đang nạp dữ liệu xe Buýt (BusBot)...")
        loop = asyncio.get_running_loop()
        models["bus_bot"] = await loop.run_in_executor(None, BusBotV13)
        logger.info("✅ BusBot Data Loaded")
    except Exception as e:
        logger.error(f"❌ Lỗi nạp BusBot: {e}")

    logger.info("🎉 SERVER ĐÃ SẴN SÀNG PHỤC VỤ! (RAM đã nạp đủ)")
    
    yield # Server chạy tại đây
    
    # Dọn dẹp khi tắt Server
    logger.info("🛑 Đang tắt Server...")
    if models["driver"]:
        await models["driver"].close()
    models.clear()

app = FastAPI(title="Travel AI Backend", lifespan=lifespan)

# Middleware & Static
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

if not os.path.exists("images_items"):
    os.makedirs("images_items")
app.mount("/images_items", StaticFiles(directory="images_items"), name="images")

# --- 3. HEALTH CHECK ---
@app.get("/")
async def health_check():
    # Kiểm tra trạng thái thực tế của các model
    status = {
        "gemini": "ok" if models["llm"] else "error",
        "vector": "ok" if models["vector"] else "error",
        "neo4j": "ok" if models["driver"] else "error",
        "bus": "ok" if models["bus_bot"] else "error"
    }
    return {"status": "alive", "details": status, "message": "Backend is ready."}

# --- 4. HELPER FUNCTIONS ---

def normalize_text(text: Union[str, List[str], None]) -> str:
    if not text: return ""
    if isinstance(text, list): text = next((t for t in text if t), "")
    return unicodedata.normalize('NFC', str(text)).lower().strip()

def clean_entity_name(raw_text: str, use_alias: bool = False) -> str:
    if not raw_text: return ""
    cleaned = raw_text.lower().strip()[:200]
    
    prefixes = [
        "lộ trình đi xe buýt từ", "lộ trình xe buýt đi từ", "lộ trình xe buýt từ", 
        "lộ trình đi từ", "lộ trình từ", "lộ trình xe buýt về", "lộ trình",
        "hướng dẫn bắt xe buýt từ", "hướng dẫn đi xe buýt từ", "hướng dẫn đón xe buýt từ",
        "hướng dẫn bắt xe từ", "hướng dẫn đi từ", "hướng dẫn đường đi từ", "hướng dẫn",
        "cách đi xe buýt từ", "cách bắt xe buýt từ", "cách đi từ", "tìm đường đi từ",
        "đường đi xe buýt từ", "đường đi từ", "làm sao để đi từ", "làm sao đi từ",
        "xe buýt đi từ", "xe buýt từ", "tuyến xe từ", "bắt xe từ", "đón xe từ", "đi xe buýt từ",
        "đi từ", "đến", "tới", "về", "sang", "qua", "tại", "ở", "khu vực", 
        "tìm đường", "chỉ đường", "cho tôi hỏi về", "thông tin về", "giới thiệu về", "biết gì về", "có gì"
    ]
    for word in prefixes:
        if cleaned.startswith(word):
            cleaned = cleaned.replace(word, "", 1).strip()
            
    suffixes = [" như thế nào", " thế nào", " ra sao", " làm sao", " ở đâu", " chỗ nào", " nhé", " nha", " nhỉ", " vậy", " ạ", " hả", " đi", " vui", " ngon", " đẹp"]
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()

    if use_alias or True:
        ENTITY_ALIASES = {
            # Địa danh du lịch
            "dinh độc lập": "hội trường thống nhất", 
            "nhà thờ đức bà": "vương cung thánh đường chính tòa đức bà sài gòn",
            "bưu điện thành phố": "bưu điện trung tâm sài gòn",
            "bến nhà rồng": "bảo tàng hồ chí minh",
            "chợ lớn": "chợ bình tây", 
            "landmart 81": "landmark 81", 
            "lăng bác": "lăng chủ tịch hồ chí minh",
            "sài gòn": "thành phố hồ chí minh",
            "hcm": "thành phố hồ chí minh",
            "tphcm": "thành phố hồ chí minh",
            
            # Trường Đại Học (Alias cho Bus)
            "đh tdt": "đại học tôn đức thắng",
            "tdt": "đại học tôn đức thắng",
            "tdtu": "đại học tôn đức thắng",
            
            "đh văn lang": "đại học văn lang",
            "vlu": "đại học văn lang",
            "văn lang": "đại học văn lang",
            "đại học văn lang": "đại học văn lang",
            
            "hutech": "đại học công nghệ thành phố hồ chí minh",
            "đh hutech": "đại học công nghệ thành phố hồ chí minh",
            
            "ueh": "đại học kinh tế thành phố hồ chí minh",
            "đh kinh tế": "đại học kinh tế thành phố hồ chí minh",
            
            "bách khoa": "đại học bách khoa",
            "hcmut": "đại học bách khoa",
            
            "sư phạm kỹ thuật": "đại học sư phạm kỹ thuật",
            "spkt": "đại học sư phạm kỹ thuật",
            
            "fpt": "đại học fpt",
            "rmit": "đại học rmit"
        }
        for alias, full_name in ENTITY_ALIASES.items():
            if alias in cleaned:
                cleaned = cleaned.replace(alias, full_name)
    
    return cleaned.title()

# --- 5. HYBRID SEARCH LOGIC ---
async def hybrid_search_tourism(text: str, province_filter: Union[str, List[str]] = None):
    # Lấy model từ biến toàn cục
    vec_model = models["vector"]
    driver = models["driver"]

    if not vec_model or not driver:
        return [] 
    
    text = clean_entity_name(text)
    search_text_norm = normalize_text(text)
    records = []
    
    async with driver.session() as session:
        p_filter_cypher = normalize_text(province_filter) if province_filter else ""
        
        # 1. Location Filter Query (Tìm theo địa điểm cụ thể trong tỉnh)
        if len(p_filter_cypher) > 2:
            cypher_loc = """
            MATCH (node:Searchable)-[:LOCATED_IN]->(p:Province)
            WHERE toLower(p.name) CONTAINS $province_norm
            AND (toLower(node.name) CONTAINS $text_norm OR toLower(node.content) CONTAINS $text_norm)
            RETURN elementId(node) as id, node, 1.5 as score, p.name as province_name, 'location_filter' as source_type LIMIT 15
            """
            try:
                r_loc = await session.run(cypher_loc, province_norm=p_filter_cypher, text_norm=search_text_norm)
                records.extend([record.data() async for record in r_loc])
            except: pass

        # 2. Vector & Keyword Query (Nếu tìm chưa đủ)
        if len(records) < 5: 
            loop = asyncio.get_running_loop()
            try:
                vector = await loop.run_in_executor(None, lambda: vec_model.encode(text).tolist())
                
                # Vector Search
                cypher_vector = "CALL db.index.vector.queryNodes('searchable_index', 50, $vector) YIELD node, score OPTIONAL MATCH (node)-[:LOCATED_IN]->(p:Province) RETURN elementId(node) as id, node, score, p.name as province_name, 'vector' as source_type"
                
                # Keyword Search (Full text fallback)
                cypher_keyword = "MATCH (node:Searchable) WHERE toLower(node.name) CONTAINS $text_norm OPTIONAL MATCH (node)-[:LOCATED_IN]->(p:Province) RETURN elementId(node) as id, node, 1.0 as score, p.name as province_name, 'keyword' as source_type LIMIT 20"
                
                try:
                    r1 = await session.run(cypher_vector, vector=vector)
                    records.extend([record.data() async for record in r1])
                except: pass
                try:
                    r2 = await session.run(cypher_keyword, text_norm=search_text_norm)
                    records.extend([record.data() async for record in r2])
                except: pass
            except: pass

    # Deduplicate & Rerank
    unique_results = {}
    for r in records:
        uid = r['id']
        # Hạ điểm nếu sai tỉnh (Soft filter)
        if province_filter:
            p_filters = province_filter if isinstance(province_filter, list) else [province_filter]
            if not any(normalize_text(pf) in normalize_text(r.get('province_name')) for pf in p_filters):
                r['score'] *= 0.1 # Phạt nặng nếu sai tỉnh
        
        # Ngưỡng vector
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
        # Lấy model từ globals
        llm = models["llm"]
        bus = models["bus_bot"]
        
        # Kiểm tra Server Ready chưa
        if not llm or not bus or not models["vector"]:
            return {"answer": "Hệ thống đang khởi động (đang nạp tài nguyên). Vui lòng đợi 10 giây rồi thử lại.", "sources": [], "images": []}

        # --- BƯỚC 1: XỬ LÝ INPUT ---
        raw_question = request.question.strip()
        if len(raw_question) > 500: return {"answer": "Câu hỏi quá dài.", "sources": [], "images": []}
        if not raw_question: return {"answer": "Bạn chưa nhập câu hỏi.", "sources": [], "images": []}

        question = re.sub(r'[\\/*.,?]+$', '', raw_question).strip()
        lower_q = question.lower()
        logger.info(f"REQ: {question}")

        # --- BƯỚC 2: SMART INTENT ROUTER ---
        # Keyword Lists
        info_keywords = ["chi tiết", "cụ thể", "rõ hơn", "thêm về", "kể về", "nói về", "biết gì", "giới thiệu", "thông tin", "review", "ăn gì", "chơi gì", "lịch sử", "có gì", "vui không"]
        hard_bus_keywords = [
            "xe bus", "xe buýt", "buýt", "buyt", "buyet", "buyết", # Bắt lỗi chính tả
            "tuyến xe", "trạm xe", "số mấy", "metro", "tàu điện",
            "cách đi", "đường đi", "làm sao đi", "đi bằng gì", "đi như thế nào", # Intent tìm đường
            "bao lâu", "đón xe", "bắt xe"
        ]
        greeting_keywords = ["xin chào", "chào", "hello", "hi", "hí", "hé lô", "hi bot", "hi ad", "alo", "cảm ơn", "thank", "hay quá", "tuyệt vời", "ok", "tạm biệt", "bye", "hi"]
        question_keywords = ["ai", "gì", "nào", "đâu", "mấy", "bao nhiêu", "sao", "thế nào"]

        has_hard_bus = any(w in lower_q for w in hard_bus_keywords)
        has_info = any(w in lower_q for w in info_keywords)
        has_greeting = any(w in lower_q for w in greeting_keywords)
        has_question_word = any(w in lower_q for w in question_keywords)

        is_bus_intent = False
        intent = "tourism" # Mặc định là du lịch

        # --- LOGIC PHÂN LOẠI MỚI ---
        if has_greeting and len(lower_q) < 50 and not has_hard_bus and not has_info and not has_question_word:
            # Chỉ Greeting khi câu ngắn VÀ KHÔNG có từ để hỏi
            intent = "greeting"
        elif has_hard_bus:
            intent = "bus"
        elif has_info or has_question_word:
            intent = "tourism"
        else:
            # Check cấu trúc tìm đường
            has_start = any(x in lower_q for x in ["từ", "đi từ", "bắt đầu từ"])
            has_end = any(x in lower_q for x in ["đến", "tới", "về", "qua"])
            is_about_topic = "về" in lower_q and not any(x in lower_q for x in ["về bến", "về trạm", "về nhà", "về đích", "về tới"])
            
            if has_start and has_end and not is_about_topic:
                intent = "bus"
        
        # Fallback LLM Router (Dùng AI để phân loại nếu keyword thất bại)
        location_filter = None
        if intent == "tourism" and not has_info and not has_greeting and len(lower_q) > 10:
            try:
                router_prompt = f"""
                Phân tích câu hỏi: "{question}".
                Yêu cầu trả về JSON: {{ "intent": "bus"|"tourism"|"greeting", "location": "tên tỉnh/thành phố nếu có" }}
                Quy tắc:
                - Hỏi về đi lại, tìm đường, xe cộ -> "bus"
                - Hỏi về địa điểm, ăn chơi, lịch sử -> "tourism"
                - Chào hỏi -> "greeting"
                """
                res = await asyncio.wait_for(asyncio.to_thread(llm.generate_content, router_prompt), timeout=3.0)
                clean = res.text.strip().replace("```json", "").replace("```", "")
                parsed = json.loads(clean)
                intent = parsed.get("intent", intent)
                location_filter = parsed.get("location")
            except: pass
            
        # Override Intent logic
        if intent == "bus" and has_info and not has_hard_bus: intent = "tourism"
        if location_filter: location_filter = clean_entity_name(str(location_filter))

        logger.info(f"🔍 FINAL INTENT: {intent} | Location: {location_filter}")

        # --- BƯỚC 3: XỬ LÝ GREETING ---
        if intent == "greeting":
            if any(x in lower_q for x in ["cảm ơn", "thank"]):
                return {"answer": "Dạ không có chi ạ! Rất vui được hỗ trợ bạn. 🥰", "sources": [], "images": []}
            elif any(x in lower_q for x in ["tạm biệt", "bye"]):
                return {"answer": "Tạm biệt! Hẹn gặp lại bạn trong chuyến đi tới. 👋", "sources": [], "images": []}
            return {"answer": "Chào bạn! Tôi là trợ lý du lịch & giao thông AI. Bạn cần tìm đường xe buýt hay gợi ý địa điểm vui chơi?", "sources": [], "images": []}

        # --- BƯỚC 4: XỬ LÝ BUS (GIAO THÔNG) ---
        if intent == "bus":
            start_loc, end_loc = None, None
            seps = [" đến ", " tới ", " về ", " sang ", " qua ", " ra "]
            prefixes = ["đi từ", "từ", "tìm đường từ", "chỉ đường từ", "đường đi từ", "lộ trình từ", "xe buýt từ", "bắt xe từ", "ghé"]
            
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
                    start_loc = clean_entity_name(start_loc)
                    end_loc = clean_entity_name(end_loc)
                    
                    loop = asyncio.get_running_loop()
                    bus_result = await loop.run_in_executor(None, lambda: bus.solve_route(start_loc, end_loc))

                    if bus_result.get("status") == "ambiguous":
                        return {
                            "answer": bus_result["message"], 
                            "options": bus_result["options"], 
                            "context_type": "bus_ambiguity",
                            "original_request": {"start": start_loc, "end": end_loc, "type": bus_result["point_type"]},
                            "sources": [], "images": []
                        }
                    if bus_result.get("status") == "error":
                        return {"answer": bus_result["message"], "sources": [], "images": []}
                    
                    path_coords = bus_result.get("path_coords", [])
                    google_link = ""
                    if path_coords:
                        s, e = path_coords[0], path_coords[-1]
                        google_link = f"https://www.google.com/maps/dir/?api=1&origin={s[1]},{s[0]}&destination={e[1]},{e[0]}&travelmode=transit"

                    polish_prompt = f"Dữ liệu lộ trình: \"\"\"{bus_result['text']}\"\"\"\nHãy viết lại hướng dẫn thật ngắn gọn, dễ hiểu. In đậm tên trạm và số xe buýt."
                    final_res = await asyncio.to_thread(llm.generate_content, polish_prompt)
                    ans = final_res.text
                    if google_link: ans += f"\n\n🔗 **[Mở Google Maps]({google_link})**"
                    return {"answer": ans, "sources": [], "images": []}
                except Exception as e:
                    logger.error(f"BUS Error: {e}")
                    return {"answer": "Xin lỗi, tôi gặp lỗi khi tìm đường. Vui lòng thử lại với tên địa điểm cụ thể hơn.", "sources": [], "images": []}
            else:
                return {"answer": "Để tìm đường, bạn vui lòng nói rõ điểm đi và điểm đến (Ví dụ: Từ Bến Thành đến Suối Tiên).", "sources": [], "images": []}

        # --- BƯỚC 5: XỬ LÝ TOURISM (VĂN HÓA - DU LỊCH) ---
        # Logic Fix: Ưu tiên trả lời rộng nếu câu hỏi rộng
        
        final_province = request.province if request.province else location_filter
        search_query = clean_entity_name(question)
        if final_province and isinstance(final_province, list): search_query = final_province[0]

        # Hybrid Search
        search_results = await hybrid_search_tourism(search_query, final_province)
        
        # Lọc ảnh
        imgs = []
        if search_results:
            imgs = list(dict.fromkeys([item['node'].get('image_url') for item in search_results if item['node'].get('image_url')]))[:2]
        
        # Xây dựng context cho RAG
        ctx = ""
        if search_results:
            ctx = "\n".join([f"- {i['node']['name']} (Tỉnh: {i.get('province_name','Unknown')}): {i['node']['content']}" for i in search_results])
        else:
            ctx = "Không tìm thấy dữ liệu cụ thể trong database."

        # PROMPT MỚI (FIX LỖI LỆCH HƯỚNG)
        rag_prompt = f"""
        VAI TRÒ: Bạn là Chuyên gia Du lịch & Văn hóa Việt Nam am hiểu sâu sắc, thân thiện.
        
        DỮ LIỆU TÌM THẤY TỪ HỆ THỐNG:
        {ctx}
        
        CÂU HỎI CỦA NGƯỜI DÙNG: "{question}"
        
        CHỈ DẪN QUAN TRỌNG:
        1. Dữ liệu hệ thống có thể bị hẹp hoặc thiếu (ví dụ: chỉ có lễ hội người Khmer). 
        2. NẾU câu hỏi mang tính tổng quát (ví dụ: "TP.HCM có gì vui?", "Ăn gì ở Hà Nội?"):
           - ĐỪNG chỉ dựa vào dữ liệu hệ thống nếu nó quá ít hoặc quá đặc thù.
           - HÃY KẾT HỢP kiến thức rộng lớn của bạn để gợi ý thêm các địa điểm nổi tiếng, món ăn đặc trưng, hoạt động phổ biến tại địa điểm đó.
           - Ví dụ: Nếu hỏi TP.HCM, hãy nhắc đến Landmark 81, Phố đi bộ, Chợ Bến Thành, Dinh Độc Lập... bên cạnh dữ liệu hệ thống.
        3. NẾU câu hỏi cụ thể (ví dụ: "Lễ hội Ok Om Bok là gì?"): Hãy bám sát dữ liệu hệ thống để trả lời chính xác.
        4. Trả lời tự nhiên, hấp dẫn, trình bày rõ ràng (dùng gạch đầu dòng).
        """
        
        res = await asyncio.to_thread(llm.generate_content, rag_prompt)
        return {"answer": res.text, "sources": search_results, "images": imgs}

    except Exception as e:
        logger.error(f"Endpoint Error: {e}")
        return JSONResponse(status_code=500, content={"answer": "Hệ thống đang bảo trì hoặc gặp sự cố nội bộ.", "sources": [], "images": []})

# --- API ẢNH ---
@app.post("/chat_with_image")
async def chat_with_image_endpoint(file: UploadFile = File(...), question: str = Form(...)):
    try:
        llm = models["llm"]
        vec_model = models["vector"]
        driver = models["driver"]

        if not llm: return {"answer": "Hệ thống AI chưa sẵn sàng.", "sources": [], "images": []}
        
        content = await file.read()
        if len(content) > 5 * 1024 * 1024: return {"answer": "Ảnh quá lớn (>5MB).", "sources": [], "images": []}
        img = Image.open(io.BytesIO(content))
        
        # Vision Identify
        vision_res = await asyncio.to_thread(llm.generate_content, ["Đây là địa điểm du lịch nào ở VN? Chỉ trả về tên địa điểm, không giải thích thêm.", img])
        detected = vision_res.text.strip()
        
        if "Unknown" in detected or len(detected) < 2:
            return {"detected_location": None, "answer": "Xin lỗi, tôi không nhận diện được địa điểm này. Bạn có thể chụp rõ hơn không?", "sources": [], "images": []}
            
        # Search & Answer
        search_results = await hybrid_search_tourism(detected)
        imgs = list(dict.fromkeys([item['node'].get('image_url') for item in search_results if item['node'].get('image_url')]))[:2]
        ctx = "\n".join([f"- {i['node']['name']}: {i['node']['content']}" for i in search_results[:3]])
        
        final_prompt = f"""
        Địa điểm trong ảnh: {detected}
        Dữ liệu hệ thống: {ctx}
        Câu hỏi người dùng: {question}
        
        Trả lời ngắn gọn, đúng trọng tâm về địa điểm này.
        """
        final = await asyncio.to_thread(llm.generate_content, final_prompt)
        return {"detected_location": detected, "answer": final.text, "sources": search_results, "images": imgs}
    except Exception as e:
        logger.error(f"Image Error: {e}")
        return {"answer": "Lỗi xử lý ảnh.", "sources": [], "images": []}

if __name__ == "__main__":
    import uvicorn
    # Hugging Face dùng port 7860, Local dùng port khác
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)