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

# --- 1. CẤU HÌNH ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# --- 2. QUẢN LÝ TÀI NGUYÊN (LAZY LOADING - CHÌA KHÓA ĐỂ FIX RENDER) ---
# Khai báo biến toàn cục là None (Chưa tải gì cả)
_llm_model = None
_vector_model = None
_neo4j_driver = None
_bus_bot = None

# Khóa an toàn để tránh 2 người cùng tải 1 lúc
_resource_lock = asyncio.Lock()

async def get_resources():
    """
    Hàm này đảm bảo trả về bộ tứ tài nguyên: (LLM, Vector, Neo4j, BusBot).
    - Nếu đã tải rồi: Trả về ngay (mất 0s).
    - Nếu chưa tải: Bắt đầu tải (người dùng đầu tiên sẽ chờ khoảng 20s).
    """
    global _llm_model, _vector_model, _neo4j_driver, _bus_bot
    
    # Kiểm tra nhanh (Fast path)
    if _llm_model and _vector_model and _neo4j_driver and _bus_bot:
        return _llm_model, _vector_model, _neo4j_driver, _bus_bot

    # Vào chế độ tải an toàn (Slow path)
    async with _resource_lock:
        # Check lại lần nữa trong lock
        if not _llm_model:
            try:
                import google.generativeai as genai
                if not GEMINI_API_KEY: raise Exception("Thiếu GEMINI_API_KEY")
                genai.configure(api_key=GEMINI_API_KEY)
                _llm_model = genai.GenerativeModel("gemini-2.0-flash")
                logger.info("✅ Gemini Loaded")
            except Exception as e: logger.error(f"Gemini Error: {e}")

        if not _neo4j_driver:
            try:
                from neo4j import AsyncGraphDatabase
                _neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
                # Verify kết nối nhẹ
                await _neo4j_driver.verify_connectivity()
                logger.info("✅ Neo4j Connected")
            except Exception as e: logger.error(f"Neo4j Error: {e}")

        if not _bus_bot:
            try:
                from bus_core import BusBotV13
                logger.info("⏳ Loading BusBot Data...")
                loop = asyncio.get_running_loop()
                _bus_bot = await loop.run_in_executor(None, BusBotV13)
                logger.info("✅ BusBot Loaded")
            except Exception as e: logger.error(f"BusBot Error: {e}")

        if not _vector_model:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("⏳ Loading Vector Model (Heavy)...")
                loop = asyncio.get_running_loop()
                _vector_model = await loop.run_in_executor(None, lambda: SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder"))
                logger.info("✅ Vector Model Loaded")
            except Exception as e: logger.error(f"Vector Model Error: {e}")
        
        return _llm_model, _vector_model, _neo4j_driver, _bus_bot

# --- 3. LIFESPAN (RỖNG TUẾCH - ĐỂ KHỞI ĐỘNG NHANH) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Render cần dòng này chạy ngay lập tức.
    logger.info("🚀 Server started instantly! Resources will load on first request.")
    yield
    # Dọn dẹp khi tắt
    logger.info("🛑 Shutting down...")
    if _neo4j_driver: await _neo4j_driver.close()

app = FastAPI(title="Travel AI", lifespan=lifespan)

# Middleware & Static
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
if not os.path.exists("images_items"): os.makedirs("images_items")
app.mount("/images_items", StaticFiles(directory="images_items"), name="images")

# --- 4. HEALTH CHECK (RENDER BẮT BUỘC PHẢI CÓ) ---
@app.get("/")
async def health_check():
    return {"status": "alive", "message": "Send POST /chat to start."}

# --- 5. HELPER FUNCTIONS (LOGIC CỦA BẠN) ---

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
        "tìm đường", "chỉ đường", "cho tôi hỏi về", "thông tin về", "giới thiệu về", "biết gì về"
    ]
    for word in prefixes:
        if cleaned.startswith(word):
            cleaned = cleaned.replace(word, "", 1).strip()
            
    suffixes = [" như thế nào", " thế nào", " ra sao", " làm sao", " ở đâu", " chỗ nào", " nhé", " nha", " nhỉ", " vậy", " ạ", " hả", " đi"]
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()

    if use_alias:
        ENTITY_ALIASES = {
            "dinh độc lập": "hội trường thống nhất", 
            "nhà thờ đức bà": "vương cung thánh đường chính tòa đức bà sài gòn",
            "bưu điện thành phố": "bưu điện trung tâm sài gòn",
            "bến nhà rồng": "bảo tàng hồ chí minh",
            "chợ lớn": "chợ bình tây", 
            "landmart 81": "landmark 81", 
            "lăng bác": "lăng chủ tịch hồ chí minh"
        }
        if cleaned in ENTITY_ALIASES: return ENTITY_ALIASES[cleaned].title()
    
    return cleaned.title()

# --- HYBRID SEARCH LOGIC (SỬA ĐỂ NHẬN MODEL TỪ NGOÀI VÀO) ---
async def hybrid_search_tourism(text: str, province_filter: Union[str, List[str]] = None, model=None, driver=None):
    if not model or not driver: return [] # Safety check
    
    text = clean_entity_name(text)
    search_text_norm = normalize_text(text)
    records = []
    
    async with driver.session() as session:
        p_filter_cypher = normalize_text(province_filter) if province_filter else ""
        
        # 1. Location Filter Query
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

        # 2. Vector & Keyword Query
        if len(records) < 5: 
            loop = asyncio.get_running_loop()
            try:
                vector = await loop.run_in_executor(None, lambda: model.encode(text).tolist())
                cypher_vector = "CALL db.index.vector.queryNodes('searchable_index', 50, $vector) YIELD node, score OPTIONAL MATCH (node)-[:LOCATED_IN]->(p:Province) RETURN elementId(node) as id, node, score, p.name as province_name, 'vector' as source_type"
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
#              MAIN API (LOGIC ĐẦY ĐỦ CỦA BẠN)
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # --- BƯỚC 1: LẤY TÀI NGUYÊN (QUAN TRỌNG) ---
        # Hàm này sẽ "treo" request đầu tiên một chút để tải model.
        # Nhưng Render sẽ KHÔNG giết app vì app đã chạy rồi.
        llm, vec_model, driver, bus = await get_resources()
        
        if not llm or not driver:
            return {"answer": "Hệ thống đang khởi động AI... Vui lòng thử lại sau 30 giây.", "sources": [], "images": []}

        # --- BƯỚC 2: XỬ LÝ INPUT ---
        raw_question = request.question.strip()
        if len(raw_question) > 500: return {"answer": "Câu hỏi quá dài.", "sources": [], "images": []}
        if not raw_question: return {"answer": "Bạn chưa nhập câu hỏi.", "sources": [], "images": []}

        question = re.sub(r'[\\/*.,?]+$', '', raw_question).strip()
        lower_q = question.lower()
        logger.info(f"REQ: {question}")

        # --- BƯỚC 3: SMART INTENT ROUTER (LOGIC CỦA BẠN) ---
        info_keywords = ["chi tiết", "cụ thể", "rõ hơn", "thêm về", "kể về", "nói về", "biết gì", "giới thiệu", "thông tin", "review", "ăn gì", "chơi gì", "lịch sử"]
        hard_bus_keywords = ["xe bus", "xe buýt", "buýt", "tuyến xe", "trạm xe", "số mấy", "metro", "tàu điện"]
        greeting_keywords = ["xin chào", "chào", "hello", "hi bot", "hi ad", "alo", "cảm ơn", "thank", "hay quá", "tuyệt vời", "ok", "tạm biệt", "bye"]

        has_hard_bus = any(w in lower_q for w in hard_bus_keywords)
        has_info = any(w in lower_q for w in info_keywords)
        has_greeting = any(w in lower_q for w in greeting_keywords)

        is_bus_intent = False
        intent = "tourism" # Default

        # Logic Ưu tiên
        if has_greeting and len(lower_q) < 50 and not has_hard_bus and not has_info:
            intent = "greeting"
        elif has_hard_bus:
            is_bus_intent = True
            intent = "bus"
        elif has_info:
            is_bus_intent = False
            intent = "tourism"
        else:
            # Check cấu trúc tìm đường (Bus)
            has_start = any(x in lower_q for x in ["từ", "đi từ", "bắt đầu từ"])
            has_end = any(x in lower_q for x in ["đến", "tới", "về", "qua"])
            is_about_topic = "về" in lower_q and not any(x in lower_q for x in ["về bến", "về trạm", "về nhà", "về đích", "về tới"])
            
            if has_start and has_end and not is_about_topic:
                is_bus_intent = True
                intent = "bus"
        
        # Fallback LLM Router
        location_filter = None
        if intent == "tourism" and not has_info and not has_greeting and len(lower_q) > 10:
            try:
                router_prompt = f"""
                Phân tích: "{question}".
                Quy tắc:
                - Có "chi tiết", "kể về" -> tourism.
                - Từ "đi" cuối câu ("kể đi") -> tourism.
                - Chỉ "bus" khi tìm đường phương tiện công cộng.
                - "cảm ơn", "chào" -> greeting.
                JSON: {{ "intent": "bus"|"tourism"|"greeting", "location": "..." }}
                """
                res = await asyncio.wait_for(asyncio.to_thread(llm.generate_content, router_prompt), timeout=3.0)
                clean = res.text.strip().replace("```json", "").replace("```", "")
                parsed = json.loads(clean)
                intent = parsed.get("intent", intent)
                location_filter = parsed.get("location")
            except: pass
            
        # Override lần cuối
        if intent == "bus" and has_info and not has_hard_bus: intent = "tourism"
        if location_filter: location_filter = clean_entity_name(str(location_filter))

        logger.info(f"🔍 FINAL INTENT: {intent}")

        # --- BƯỚC 4: XỬ LÝ GREETING ---
        if intent == "greeting":
            if any(x in lower_q for x in ["cảm ơn", "thank", "hay quá", "tuyệt"]):
                return {"answer": "Dạ không có chi ạ! Rất vui được hỗ trợ bạn. 🥰", "sources": [], "images": []}
            elif any(x in lower_q for x in ["tạm biệt", "bye"]):
                return {"answer": "Tạm biệt! Hẹn gặp lại bạn. 👋", "sources": [], "images": []}
            return {"answer": "Chào bạn! Tôi là trợ lý AI. Bạn cần tìm đường xe buýt hay thông tin du lịch?", "sources": [], "images": []}

        # --- BƯỚC 5: XỬ LÝ BUS ---
        if intent == "bus":
            if not bus: return {"answer": "Dữ liệu xe buýt chưa sẵn sàng (đang tải). Thử lại sau 10s nhé."}
            
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

                    polish_prompt = f"Dữ liệu: \"\"\"{bus_result['text']}\"\"\"\nViết lại hướng dẫn ngắn gọn, dễ hiểu. In đậm tên trạm và số xe. Hiển thị giá vé."
                    final_res = await asyncio.to_thread(llm.generate_content, polish_prompt)
                    ans = final_res.text
                    if google_link: ans += f"\n\n🔗 **[Mở Google Maps]({google_link})**"
                    return {"answer": ans, "sources": [], "images": []}
                except Exception as e:
                    logger.error(f"BUS Error: {e}")
                    return {"answer": "Lỗi xử lý tìm đường. Vui lòng thử lại.", "sources": [], "images": []}
            else:
                return {"answer": "Vui lòng cung cấp điểm đi và điểm đến (Ví dụ: Từ Bến Thành đến Suối Tiên)", "sources": [], "images": []}

        # --- BƯỚC 6: XỬ LÝ TOURISM ---
        final_province = request.province if request.province else location_filter
        search_query = clean_entity_name(question)
        if final_province and isinstance(final_province, list): search_query = final_province[0]

        # Hybrid Search (Truyền model & driver vào)
        search_results = await hybrid_search_tourism(search_query, final_province, model=vec_model, driver=driver)
        
        if search_results:
            imgs = list(dict.fromkeys([item['node'].get('image_url') for item in search_results if item['node'].get('image_url')]))[:2]
            ctx = "\n".join([f"- {i['node']['name']} (Tỉnh: {i.get('province_name','Unknown')}): {i['node']['content']}" for i in search_results])
            
            rag_prompt = f"""
            VAI TRÒ: Chuyên gia Văn hóa - Du lịch Việt Nam.
            DỮ LIỆU: {ctx}
            CÂU HỎI: "{question}"
            YÊU CẦU: Trả lời tự nhiên, hấp dẫn, đúng trọng tâm.
            """
            res = await asyncio.to_thread(llm.generate_content, rag_prompt)
            return {"answer": res.text, "sources": search_results, "images": imgs}
        else:
            safe_prompt = f"Bạn là trợ lý du lịch VN. Câu hỏi: '{question}'. Trả lời ngắn gọn nếu biết. Nếu hỏi về Code/Y tế/Tài chính -> Từ chối khéo."
            res = await asyncio.to_thread(llm.generate_content, safe_prompt)
            return {"answer": res.text, "sources": [], "images": []}

    except Exception as e:
        logger.error(f"Endpoint Error: {e}")
        return JSONResponse(status_code=500, content={"answer": "Lỗi hệ thống nội bộ.", "sources": [], "images": []})

# --- API ẢNH (DÙNG LAZY LOADER) ---
@app.post("/chat_with_image")
async def chat_with_image_endpoint(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # Load tài nguyên
        llm, vec_model, driver, _ = await get_resources()
        if not llm: return {"answer": "Hệ thống AI chưa sẵn sàng.", "sources": [], "images": []}
        
        content = await file.read()
        if len(content) > 5 * 1024 * 1024: return {"answer": "Ảnh quá lớn (>5MB).", "sources": [], "images": []}
        img = Image.open(io.BytesIO(content))
        
        # Vision
        vision_res = await asyncio.to_thread(llm.generate_content, ["Đây là địa điểm du lịch nào ở VN? Chỉ trả về tên.", img])
        detected = vision_res.text.strip()
        
        if "Unknown" in detected or len(detected) < 2:
            return {"detected_location": None, "answer": "Không nhận diện được địa điểm.", "sources": [], "images": []}
            
        # Search & Answer
        search_results = await hybrid_search_tourism(detected, model=vec_model, driver=driver)
        imgs = list(dict.fromkeys([item['node'].get('image_url') for item in search_results if item['node'].get('image_url')]))[:2]
        ctx = "\n".join([f"- {i['node']['name']}: {i['node']['content']}" for i in search_results[:3]])
        
        final = await asyncio.to_thread(llm.generate_content, f"Địa điểm: {detected}\nDữ liệu: {ctx}\nHỏi: {question}\nTrả lời:")
        return {"detected_location": detected, "answer": final.text, "sources": search_results, "images": imgs}
    except Exception as e:
        return {"answer": "Lỗi xử lý ảnh.", "sources": [], "images": []}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)