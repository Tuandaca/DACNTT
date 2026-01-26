import os
import logging
import asyncio
import unicodedata
import io
import json
import time
import re
from typing import Optional, Union, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from neo4j import AsyncGraphDatabase
import google.generativeai as genai
from PIL import Image

# Import Bus Core (Đảm bảo file bus_core.py nằm cùng thư mục)
from bus_core import BusBotV13

from fastapi.middleware.cors import CORSMiddleware

# --- 1. CẤU HÌNH LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# --- 2. LOAD BIẾN MÔI TRƯỜNG ---
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# --- 3. QUẢN LÝ TÀI NGUYÊN (LAZY LOADING + LOCK) ---
llm_model = None
vector_model = None
tourism_driver = None
bus_bot = None

# Khóa an toàn để tránh Race Condition khi nhiều người cùng gọi load model
model_lock = asyncio.Lock()

async def get_vector_model():
    global vector_model
    async with model_lock: # Chỉ 1 luồng được tải tại 1 thời điểm
        if vector_model is None:
            logger.info("⏳ SYSTEM: Đang tải BKAI Vector Model...")
            try:
                # Chạy trong thread pool để không chặn event loop
                loop = asyncio.get_running_loop()
                vector_model = await loop.run_in_executor(None, lambda: SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder"))
                logger.info("✅ Vector Model Loaded.")
            except Exception as e:
                logger.error(f"❌ Lỗi tải Vector Model: {e}")
                raise e
    return vector_model

async def get_bus_bot():
    global bus_bot
    async with model_lock:
        if bus_bot is None:
            logger.info("⏳ SYSTEM: Đang khởi tạo BusBot...")
            try:
                loop = asyncio.get_running_loop()
                bus_bot = await loop.run_in_executor(None, BusBotV13)
                logger.info("✅ BusBot Ready.")
            except Exception as e:
                logger.error(f"❌ Lỗi khởi tạo BusBot: {e}")
                raise e
    return bus_bot

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_model, tourism_driver
    logger.info("🚀 SYSTEM: Server đang khởi động...")
    
    # Cấu hình Gemini
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        llm_model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("✅ Gemini Configured.")
    else:
        logger.warning("⚠️ Thiếu GEMINI_API_KEY!")

    # Kết nối Neo4j
    try:
        tourism_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        await tourism_driver.verify_connectivity() 
        logger.info("✅ Neo4j Connected.")
    except Exception as e:
        logger.error(f"❌ Lỗi kết nối Neo4j: {e}")
    
    yield
    
    logger.info("🛑 SYSTEM: Đang tắt server...")
    if tourism_driver:
        await tourism_driver.close()

app = FastAPI(title="Professional Travel & Transport AI", lifespan=lifespan)

# CORS & Static Files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
if not os.path.exists("images_items"): os.makedirs("images_items")
app.mount("/images_items", StaticFiles(directory="images_items"), name="images")

# --- HELPER FUNCTIONS ---
def normalize_text(text: Union[str, List[str], None]) -> str:
    if not text: return ""
    if isinstance(text, list): text = next((t for t in text if t), "")
    return unicodedata.normalize('NFC', str(text)).lower().strip()

def clean_entity_name(raw_text: str, use_alias: bool = False) -> str:
    if not raw_text: return ""
    cleaned = raw_text.lower().strip()[:200] # Cắt ngắn nếu quá dài
    
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

# --- HYBRID SEARCH LOGIC ---
async def hybrid_search_tourism(text: str, province_filter: Union[str, List[str]] = None):
    text = clean_entity_name(text)
    search_text_norm = normalize_text(text)
    records = []
    
    # Lấy model an toàn
    current_model = await get_vector_model()
    
    async with tourism_driver.session() as session:
        p_filter_cypher = normalize_text(province_filter) if province_filter else ""

        # Query 1: Filter
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
            except Exception as e: logger.error(f"Error Loc Search: {e}")

        # Query 2: Vector & Keyword
        if len(records) < 5: 
            loop = asyncio.get_running_loop()
            try:
                vector = await loop.run_in_executor(None, lambda: current_model.encode(text).tolist())
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
            except Exception as e: logger.error(f"Vector Error: {e}")

    # Deduplicate & Rerank
    unique_results = {}
    for r in records:
        uid = r['id']
        if province_filter:
            # Logic check tỉnh
            p_filters = province_filter if isinstance(province_filter, list) else [province_filter]
            if not any(normalize_text(pf) in normalize_text(r.get('province_name')) for pf in p_filters):
                r['score'] *= 0.1
        
        if r['source_type'] == 'vector' and r['score'] < 0.65: continue
        
        if uid not in unique_results:
            unique_results[uid] = r
            unique_results[uid]['score'] = 10.0 if r['source_type'] in ['keyword', 'location_filter'] else r['score']
            
    return sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)[:10]

# --- REQUEST MODEL ---
class ChatRequest(BaseModel):
    question: str
    province: Optional[Union[str, List[str]]] = None

# ==========================================
#              MAIN API
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 0. GLOBAL ERROR HANDLING
    try:
        # Check System Health
        if not llm_model or not tourism_driver:
            return JSONResponse(status_code=503, content={"answer": "Hệ thống đang khởi động AI (mất khoảng 30s). Vui lòng thử lại sau.", "sources": [], "images": []})

        # 1. INPUT SANITIZATION
        raw_question = request.question.strip()
        if len(raw_question) > 500: 
            return {"answer": "Câu hỏi quá dài. Vui lòng hỏi ngắn gọn dưới 500 ký tự.", "sources": [], "images": []}
        if not raw_question:
            return {"answer": "Bạn chưa nhập câu hỏi.", "sources": [], "images": []}

        question = re.sub(r'[\\/*.,?]+$', '', raw_question).strip()
        lower_q = question.lower()
        logger.info(f"REQ: {question}")

        # 2. SMART INTENT ROUTER
        is_bus_intent = False
        
        # Nhóm từ khóa hỏi thông tin (Gặp là Tourism ngay)
        info_keywords = ["chi tiết", "cụ thể", "rõ hơn", "thêm về", "kể về", "nói về", "biết gì", "giới thiệu", "thông tin", "review", "ăn gì", "chơi gì", "lịch sử"]
        # Nhóm từ khóa xe buýt cứng
        hard_bus_keywords = ["xe bus", "xe buýt", "buýt", "tuyến xe", "trạm xe", "số mấy", "metro", "tàu điện"]
        # Từ khóa định tuyến (Phải có cấu trúc)
        routing_markers = ["từ", "đến", "tới", "về", "qua"]

        has_hard_bus = any(w in lower_q for w in hard_bus_keywords)
        has_info = any(w in lower_q for w in info_keywords)

        if has_hard_bus:
            is_bus_intent = True
        elif has_info:
            is_bus_intent = False # Kể cả có chữ "đi" ở cuối, nếu có "chi tiết" thì là Tourism
        else:
            # Check cấu trúc "Từ... Đến..."
            has_start = any(x in lower_q for x in ["từ", "đi từ", "bắt đầu từ"])
            has_end = any(x in lower_q for x in ["đến", "tới", "về", "qua"])
            
            # Lọc nhiễu từ "về" (về chủ đề vs về nhà)
            is_about_topic = "về" in lower_q and not any(x in lower_q for x in ["về bến", "về trạm", "về nhà", "về đích", "về tới"])
            
            if has_start and has_end and not is_about_topic:
                is_bus_intent = True
            
        # Fallback LLM Router (Nếu vẫn chưa chắc)
        intent = "bus" if is_bus_intent else "tourism"
        location_filter = None
        mode = "detail"

        if not has_hard_bus and not has_info:
            try:
                # Timeout cho Router là 3 giây để không làm chậm App
                router_prompt = f"""
                Phân tích: "{question}".
                Quy tắc:
                - Có "chi tiết", "kể về" -> tourism.
                - Từ "đi" cuối câu ("kể đi") -> tourism.
                - Chỉ "bus" khi tìm đường phương tiện công cộng.
                JSON: {{ "intent": "bus"|"tourism"|"greeting"|"ood", "location": "...", "mode": "detail"|"list" }}
                """
                res = await asyncio.wait_for(asyncio.to_thread(llm_model.generate_content, router_prompt), timeout=3.0)
                clean = res.text.strip().replace("```json", "").replace("```", "")
                parsed = json.loads(clean)
                intent = parsed.get("intent", intent)
                location_filter = parsed.get("location")
                mode = parsed.get("mode", "detail")
            except: pass # Nếu lỗi hoặc timeout thì dùng logic if-else bên trên

        # Override lần cuối để chắc ăn
        if intent == "bus" and has_info and not has_hard_bus: intent = "tourism"
        if location_filter: location_filter = clean_entity_name(str(location_filter))

        logger.info(f"🔍 FINAL INTENT: {intent} | LOC: {location_filter}")

        if intent == "greeting":
            return {"answer": "Chào bạn! Tôi là trợ lý AI. Bạn cần tìm đường xe buýt hay thông tin du lịch?", "sources": [], "images": []}

        # 3. XỬ LÝ BUS
        if intent == "bus":
            current_bus_bot = await get_bus_bot() # Async get
            
            # Trích xuất điểm đi/đến (Logic tách chuỗi)
            start_loc, end_loc = None, None
            seps = [" đến ", " tới ", " về ", " sang ", " qua ", " ra "]
            prefixes = ["đi từ", "từ", "tìm đường từ", "chỉ đường từ", "đường đi từ", "lộ trình từ", "xe buýt từ"]
            
            # Xử lý cắt chuỗi
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
                    
                    # Gọi BusBot (Chạy trong thread pool để không chặn)
                    loop = asyncio.get_running_loop()
                    bus_result = await loop.run_in_executor(None, lambda: current_bus_bot.solve_route(start_loc, end_loc))

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

                    # Polish lời giải bằng LLM
                    path_coords = bus_result.get("path_coords", [])
                    google_link = ""
                    if path_coords:
                        s, e = path_coords[0], path_coords[-1]
                        google_link = f"https://www.google.com/maps/dir/?api=1&origin={s[1]},{s[0]}&destination={e[1]},{e[0]}&travelmode=transit"

                    polish_prompt = f"Dữ liệu: \"\"\"{bus_result['text']}\"\"\"\nViết lại hướng dẫn ngắn gọn, dễ hiểu. In đậm trạm và xe."
                    final_res = await asyncio.to_thread(llm_model.generate_content, polish_prompt)
                    
                    ans = final_res.text
                    if google_link: ans += f"\n\n🔗 **[Mở Google Maps]({google_link})**"
                    return {"answer": ans, "sources": [], "images": []}
                except Exception as e:
                    logger.error(f"Bus Logic Error: {e}")
                    return {"answer": "Lỗi xử lý tìm đường. Vui lòng thử lại.", "sources": [], "images": []}
            else:
                return {"answer": "Bạn muốn đi từ đâu đến đâu? (Ví dụ: Từ Bến Thành đến Suối Tiên)", "sources": [], "images": []}

        # 4. XỬ LÝ OOD (Out of Domain)
        if intent == "ood":
            # Fallback nhanh
            return {"answer": "Tôi chỉ hỗ trợ thông tin Du lịch và Xe buýt TP.HCM. Vui lòng hỏi đúng chủ đề.", "sources": [], "images": []}

        # 5. XỬ LÝ TOURISM
        final_province = request.province if request.province else location_filter
        search_query = clean_entity_name(question)
        if final_province and isinstance(final_province, list): search_query = final_province[0]

        search_results = await hybrid_search_tourism(search_query, final_province)
        
        if search_results:
            imgs = list(dict.fromkeys([item['node'].get('image_url') for item in search_results if item['node'].get('image_url')]))[:2]
            ctx = "\n".join([f"- {i['node']['name']}: {i['node']['content']}" for i in search_results])
            
            rag_prompt = f"""
            VAI TRÒ: HDV Du lịch.
            DỮ LIỆU: {ctx}
            CÂU HỎI: "{question}"
            YÊU CẦU: Trả lời tự nhiên, hấp dẫn. Nếu là dạng liệt kê thì dùng gạch đầu dòng.
            """
            res = await asyncio.to_thread(llm_model.generate_content, rag_prompt)
            return {"answer": res.text, "sources": search_results, "images": imgs}
        else:
            # Fallback kiến thức chung (Chống leak)
            safe_prompt = f"Bạn là trợ lý du lịch VN. Câu hỏi: '{question}'. Trả lời ngắn gọn nếu biết. Nếu hỏi về Code/Y tế/Tài chính -> Từ chối."
            res = await asyncio.to_thread(llm_model.generate_content, safe_prompt)
            return {"answer": res.text, "sources": [], "images": []}

    except Exception as e:
        logger.error(f"CRITICAL SERVER ERROR: {e}")
        return JSONResponse(status_code=200, content={"answer": "Hệ thống đang bảo trì hoặc gặp lỗi không mong muốn.", "sources": [], "images": []})

# --- IMAGE API ---
@app.post("/chat_with_image")
async def chat_with_image_endpoint(file: UploadFile = File(...), question: str = Form(...)):
    try:
        if not llm_model: return {"answer": "Server đang khởi động...", "sources": [], "images": []}
        
        content = await file.read()
        if len(content) > 5 * 1024 * 1024: # Limit 5MB
            return {"answer": "Ảnh quá lớn (>5MB). Vui lòng chọn ảnh nhỏ hơn.", "sources": [], "images": []}
            
        img = Image.open(io.BytesIO(content))
        
        vision_res = await asyncio.to_thread(llm_model.generate_content, ["Đây là địa điểm du lịch nào ở VN? Chỉ trả về tên.", img])
        detected = vision_res.text.strip()
        
        if "Unknown" in detected or len(detected) < 2:
            return {"detected_location": None, "answer": "Không nhận diện được địa điểm.", "sources": [], "images": []}
            
        search_results = await hybrid_search_tourism(detected)
        imgs = list(dict.fromkeys([item['node'].get('image_url') for item in search_results if item['node'].get('image_url')]))[:2]
        
        ctx = "\n".join([f"- {i['node']['name']}: {i['node']['content']}" for i in search_results[:3]])
        prompt = f"Địa điểm: {detected}\nDữ liệu: {ctx}\nHỏi: {question}\nTrả lời:"
        
        final = await asyncio.to_thread(llm_model.generate_content, prompt)
        return {"detected_location": detected, "answer": final.text, "sources": search_results, "images": imgs}

    except Exception as e:
        logger.error(f"Image Error: {e}")
        return {"answer": "Lỗi xử lý ảnh.", "sources": [], "images": []}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)