import os
import logging
import asyncio
import unicodedata
import io
import json
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from neo4j import AsyncGraphDatabase
import google.generativeai as genai
from PIL import Image

from bus_core import BusBotV13

# --- 1. CẤU HÌNH LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# --- 2. LOAD BIẾN MÔI TRƯỜNG ---
load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# --- 3. KHỞI TẠO MODEL & DB ---
if not GEMINI_API_KEY:
    logger.error("❌ CHƯA CÓ GEMINI_API_KEY TRONG FILE .ENV")

genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel('gemini-2.0-flash') 

logger.info("SYSTEM: Loading Vector Model...")
model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
logger.info("SYSTEM: Vector Model Loaded.")

tourism_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
logger.info("SYSTEM: Initializing BusBot...")
bus_bot = BusBotV13()
logger.info("SYSTEM: BusBot Ready.")

def normalize_text(text: str) -> str:
    if not text: return ""
    return unicodedata.normalize('NFC', text).lower().strip()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await tourism_driver.close()
    bus_bot.close()

app = FastAPI(title="Professional Travel & Transport AI", lifespan=lifespan)

if not os.path.exists("images_items"):
    os.makedirs("images_items")
app.mount("/images_items", StaticFiles(directory="images_items"), name="images")

# --- 4. DANH SÁCH TỪ KHÓA ---
QUESTION_MARKERS = [
    "?", "là gì", "cái gì", "việc gì", "thứ gì", "điều gì",
    "ở đâu", "chỗ nào", "nơi nào", "khu nào", "đâu",
    "khi nào", "lúc nào", "bao giờ", "hồi nào", "ngày nào",
    "ai", "người nào", "nhân vật nào", "vị nào",
    "tại sao", "vì sao", "nguyên nhân", "lý do", "sao lại",
    "thế nào", "ra sao", "như nào", "làm sao", "cách nào",
    "bao nhiêu", "bao lâu", "bao xa", "mấy", "số lượng",
    "có...không", "được không", "có phải", "không ạ", "chưa", "hả", "hử",
    "cho hỏi", "muốn hỏi", "tìm hiểu", "chỉ giúp", "hướng dẫn", "tư vấn",
    "thông tin", "chi tiết", "giới thiệu", "kể về", "nói về", "liệt kê", "danh sách"
]

# --- 5. LOGIC TÌM KIẾM TOURISM ---
async def hybrid_search_tourism(text: str, province_filter: str = None):
    search_text_norm = normalize_text(text)
    records = []
    
    async with tourism_driver.session() as session:
        # TH1: Có lọc theo tỉnh -> Tìm kiếm diện rộng
        if province_filter and len(province_filter) > 2:
            province_norm = normalize_text(province_filter)
            cypher_loc = """
            MATCH (node:Searchable)-[:LOCATED_IN]->(p:Province)
            WHERE toLower(p.name) CONTAINS $province_norm
            AND (toLower(node.name) CONTAINS $text_norm OR toLower(node.content) CONTAINS $text_norm)
            RETURN elementId(node) as id, node, 1.5 as score, p.name as province_name, 'location_filter' as source_type
            LIMIT 15
            """
            try:
                r_loc = await session.run(cypher_loc, province_norm=province_norm, text_norm=search_text_norm)
                records.extend([record.data() async for record in r_loc])
            except Exception as e:
                logger.error(f"Error Loc Search: {e}")

        # TH2: Vector & Keyword Search
        if len(records) < 5: 
            loop = asyncio.get_running_loop()
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

    unique_results = {}
    for r in records:
        uid = r['id']
        if province_filter and r.get('province_name'):
            if normalize_text(province_filter) not in normalize_text(r['province_name']):
                r['score'] = r['score'] * 0.1
        if r['source_type'] == 'vector' and r['score'] < 0.65: continue
        
        if uid not in unique_results:
            unique_results[uid] = r
            if r['source_type'] == 'location_filter':
                unique_results[uid]['score'] = 10.0
            else:
                unique_results[uid]['score'] = 10.0 if r['source_type'] == 'keyword' else r['score']
    
    sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
    return sorted_results[:10]

# --- 6. FALLBACK LOGIC ---
async def fallback_general_knowledge(question: str):
    system_prompt = f"""
    VAI TRÒ: Chuyên gia tư vấn du lịch và văn hóa Việt Nam.
    CÂU HỎI: "{question}"
    YÊU CẦU: Trả lời chính xác, khách quan, ngôn ngữ trang trọng. KHÔNG dùng Emoji.
    """
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, lambda: llm_model.generate_content(system_prompt))
    return response.text

class ChatRequest(BaseModel):
    question: str
    province: Optional[str] = None

# ==========================================
#               MAIN CHAT API
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    question = request.question.strip()
    logger.info(f"REQ: {question}")
    lower_q = question.lower()
    
    # --- 1. ROUTER THÔNG MINH ---
    # Danh sách từ nối để nhận diện câu hỏi Bus
    # Bao gồm: đến, tới, về, sang, qua, ra
    dest_markers = [" đến ", " tới ", " về ", " sang ", " qua ", " ra "] 
    
    is_bus_intent = False
    # Check nhanh: Nếu có từ chỉ hướng + từ chỉ di chuyển
    if any(m.strip() in lower_q for m in dest_markers) and \
       any(w in lower_q for w in ["từ", "đi", "đường", "xe bus", "buýt", "cách"]):
        is_bus_intent = True

    router_prompt = f"""
    Phân tích câu hỏi: "{question}".
    Nhiệm vụ:
    1. Intent: "greeting" | "bus" | "tourism".
    2. Location: Trích xuất tên Tỉnh/Thành phố (nếu có).
    3. Mode: "list" (liệt kê) | "detail" (chi tiết).
    
    JSON Output: {{ "intent": "...", "location": "...", "mode": "..." }}
    """
    
    intent = "bus" if is_bus_intent else "tourism"
    location_filter = None
    mode = "detail"

    try:
        if not is_bus_intent: # Chỉ dùng AI check nếu chưa chắc là Bus
            res = await asyncio.to_thread(llm_model.generate_content, router_prompt)
            clean = res.text.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(clean)
            intent = parsed.get("intent", "tourism")
            location_filter = parsed.get("location")
            mode = parsed.get("mode", "detail")
    except: pass

    # Override Intent thủ công nếu phát hiện từ khóa mạnh
    if is_bus_intent: intent = "bus"
    
    has_question_word = any(marker in lower_q for marker in QUESTION_MARKERS)
    if intent != "bus":
        if has_question_word:
            if intent == "greeting": intent = "tourism"
        else:
            greeting_words = ["xin chào", "hello", "hi bot", "chào bạn", "alo"]
            if any(w == lower_q or lower_q.startswith(w) for w in greeting_words) and len(lower_q) < 20:
                intent = "greeting"

    if location_filter and mode == "detail":
        if any(w in lower_q for w in ["nào", "gì", "những", "các"]): mode = "list"

    logger.info(f"🔍 INTENT: {intent} | LOC: {location_filter} | MODE: {mode}")

    if intent == "greeting":
        return {"answer": "Kính chào Quý khách. Tôi là Trợ lý AI chuyên trách về Văn hóa & Giao thông.\nTôi có thể giúp bạn tra cứu lộ trình xe buýt hoặc thông tin du lịch.", "sources": [], "images": []}

    # --- XỬ LÝ BUS (LOGIC TÁCH CHUỖI NÂNG CAO) ---
    if intent == "bus":
        start_loc = None
        end_loc = None
        
        # Danh sách các từ có thể là vách ngăn giữa Điểm đi và Điểm đến
        # Lưu ý: Có khoảng trắng 2 đầu để tránh bắt nhầm (vd: "Quốc lộ" có chữ "qua")
        separators = [" đến ", " tới ", " về ", " sang ", " qua ", " ra "]
        
        # Danh sách từ thừa ở đầu câu cần cắt bỏ
        start_prefixes = ["đi từ", "từ", "tìm đường từ", "chỉ đường từ", "đường đi từ", "lộ trình từ", "xe buýt từ", "bắt xe từ"]

        found_sep = None
        # Tìm từ nối xuất hiện đầu tiên trong câu
        for sep in separators:
            if sep in lower_q:
                found_sep = sep
                break
        
        if found_sep:
            try:
                parts = lower_q.split(found_sep, 1) # Chỉ tách ở từ nối đầu tiên
                end_loc = parts[1].strip()
                
                # Xử lý phần Start: Cắt bỏ các từ prefix
                start_raw = parts[0].strip()
                for prefix in start_prefixes:
                    if start_raw.startswith(prefix):
                        start_raw = start_raw[len(prefix):].strip()
                        break # Cắt xong thì thôi
                start_loc = start_raw
            except: pass
        
        if start_loc and end_loc:
            try:
                # Gọi BusBot
                bus_result = await asyncio.to_thread(bus_bot.solve_route, start_loc, end_loc)
                
                # TH1: Ambiguous -> Trả về Options
                if bus_result.get("status") == "ambiguous":
                    return {
                        "answer": bus_result["message"], 
                        "options": bus_result["options"],
                        "context_type": "bus_ambiguity",
                        "original_request": {"start": start_loc, "end": end_loc, "type": bus_result["point_type"]},
                        "sources": [], "images": []
                    }

                # TH2: Error
                if bus_result.get("status") == "error":
                    return {"answer": bus_result["message"], "sources": [], "images": []}
                
                # TH3: Thành công
                raw_text = bus_result["text"]
                path_coords = bus_result.get("path_coords", [])
                
                google_link = ""
                if path_coords:
                    s = path_coords[0]
                    e = path_coords[-1]
                    google_link = f"https://www.google.com/maps/dir/?api=1&origin={s[1]},{s[0]}&destination={e[1]},{e[0]}&travelmode=transit"

                polish_prompt = f"""
                Dữ liệu lộ trình: \"\"\"{raw_text}\"\"\"
                YÊU CẦU: Viết lại hướng dẫn di chuyển chuyên nghiệp.
                1. Trình bày các bước rõ ràng.
                2. In đậm tên trạm, số xe.
                3. Bắt buộc hiển thị Bảng giá vé (Gồm: Vé thường, HSSV, Người cao tuổi).
                4. Văn phong lịch sự, không emoji.
                """
                final_res = await asyncio.to_thread(llm_model.generate_content, polish_prompt)
                
                answer_text = final_res.text
                if google_link:
                    answer_text += f"\n\n🔗 **[Xem bản đồ lộ trình trên Google Maps]({google_link})**"

                return {"answer": answer_text, "sources": [], "images": []}
            except Exception as e:
                return {"answer": f"Lỗi hệ thống: {str(e)}", "sources": [], "images": []}
        else:
             return {"answer": "Vui lòng cung cấp điểm đi và điểm đến (Ví dụ: Từ Bến Thành tới Aeon Mall) để tôi tìm lộ trình.", "sources": [], "images": []}

    # --- XỬ LÝ TOURISM ---
    final_province = request.province if request.province else location_filter
    search_results = await hybrid_search_tourism(question, final_province)
    
    if search_results:
        image_urls = []
        for item in search_results:
            node_data = item.get('node', {})
            url = node_data.get('image_url')
            if url and isinstance(url, str) and url.startswith("http"):
                image_urls.append(url)
        final_imgs = list(dict.fromkeys(image_urls))[:2] 

        context_str = "\n".join([f"- {item['node']['name']} (Tỉnh: {item.get('province_name', 'Chưa rõ')}): {item['node']['content']}" for item in search_results])
        
        if mode == "list":
            rag_prompt = f"""
            VAI TRÒ: Trợ lý du lịch.
            NHIỆM VỤ: Liệt kê địa điểm.
            DỮ LIỆU: {context_str}
            CÂU HỎI: "{question}"
            YÊU CẦU: Liệt kê danh sách (tối thiểu 3). Ghi rõ Tên và Tỉnh/Thành phố. Mô tả ngắn. KHÔNG emoji.
            """
        else:
            rag_prompt = f"""
            VAI TRÒ: Chuyên gia văn hóa & du lịch.
            DỮ LIỆU: {context_str}
            CÂU HỎI: "{question}"
            YÊU CẦU:
            1. PHONG CÁCH: Ngắn gọn, súc tích.
            2. CẤU TRÚC:
               - Đoạn 1: Tổng quan (Tên, Vị trí hành chính cụ thể Tỉnh/Thành, đặc điểm).
               - Đoạn 2: Gạch đầu dòng tóm tắt (Vị trí, Lịch sử, Kiến trúc, Giá trị).
            3. Dùng thông tin Tỉnh/Thành có trong dữ liệu. KHÔNG mở bài/kết bài rườm rà.
            """

        res = await asyncio.to_thread(llm_model.generate_content, rag_prompt)
        return {"answer": res.text, "sources": search_results, "images": final_imgs}
    
    else:
        general_answer = await fallback_general_knowledge(question)
        return {"answer": general_answer, "sources": [], "images": []}

# --- API XỬ LÝ ẢNH ---
@app.post("/chat_with_image")
async def chat_with_image_endpoint(file: UploadFile = File(...), question: str = Form(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        vision_prompt = "Đây là địa điểm nào? Chỉ trả về tên chính xác. Nếu không biết, trả về 'Unknown'."
        loop = asyncio.get_running_loop()
        vision_res = await loop.run_in_executor(None, lambda: llm_model.generate_content([vision_prompt, image]))
        detected_name = vision_res.text.strip()
        
        if "Unknown" in detected_name or len(detected_name) < 2:
            return {"detected_location": None, "answer": "Chưa nhận diện được địa điểm.", "sources": [], "images": []}
             
        search_results = await hybrid_search_tourism(detected_name)
        
        image_urls = []
        if search_results:
            for item in search_results:
                url = item.get('node', {}).get('image_url')
                if url and isinstance(url, str) and url.startswith("http"): 
                    image_urls.append(url)
        final_imgs = list(dict.fromkeys(image_urls))[:2]

        if search_results:
            context_str = "\n".join([f"- {item['node']['name']} (Tỉnh: {item.get('province_name', '')}): {item['node']['content']}" for item in search_results[:3]])
            final_prompt = f"""
            VAI TRÒ: Chuyên gia văn hóa.
            ĐỊA ĐIỂM TỪ ẢNH: {detected_name}
            DỮ LIỆU: {context_str}
            CÂU HỎI: {question}
            YÊU CẦU: Trả lời chuyên nghiệp, cấu trúc rõ ràng (Tên, Vị trí Tỉnh/Thành, Đặc điểm), KHÔNG emoji.
            """
            final_res = await loop.run_in_executor(None, lambda: llm_model.generate_content(final_prompt))
            return {"detected_location": detected_name, "answer": final_res.text, "sources": search_results, "images": final_imgs}
        else:
            general_prompt = f"Địa điểm trong ảnh là '{detected_name}'. Câu hỏi: '{question}'. Trả lời chi tiết, nêu rõ vị trí thuộc Tỉnh/Thành nào. KHÔNG emoji."
            final_res = await loop.run_in_executor(None, lambda: llm_model.generate_content(general_prompt))
            return {"detected_location": detected_name, "answer": final_res.text, "sources": [], "images": []}

    except Exception as e:
        logger.error(f"ERROR: {e}")
        return {"answer": "Lỗi xử lý ảnh.", "sources": [], "images": []}