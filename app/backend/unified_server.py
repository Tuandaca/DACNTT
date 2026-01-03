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

# Mount thư mục ảnh tĩnh (dù ưu tiên dùng link online, vẫn giữ cấu hình này để tránh lỗi nếu có request tới)
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

# --- 5. LOGIC TÌM KIẾM TOURISM (Lấy image_url) ---
async def hybrid_search_tourism(text: str, province_filter: str = None):
    search_text_norm = normalize_text(text)
    records = []
    
    async with tourism_driver.session() as session:
        # TH1: Có lọc theo tỉnh -> Tìm kiếm diện rộng (List)
        if province_filter and len(province_filter) > 2:
            province_norm = normalize_text(province_filter)
            # Tăng LIMIT để lấy nhiều địa điểm cho việc liệt kê
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

        # TH2: Vector Search & Keyword Search (Bổ trợ nếu ít kết quả)
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

    # Lọc trùng và chấm điểm lại
    unique_results = {}
    for r in records:
        uid = r['id']
        # Giảm điểm nếu sai tỉnh (Strict filtering logic)
        if province_filter and r.get('province_name'):
            if normalize_text(province_filter) not in normalize_text(r['province_name']):
                r['score'] = r['score'] * 0.1

        # Ngưỡng vector (để loại bỏ kết quả rác)
        if r['source_type'] == 'vector' and r['score'] < 0.65: continue
        
        if uid not in unique_results:
            unique_results[uid] = r
            if r['source_type'] == 'location_filter':
                unique_results[uid]['score'] = 10.0
            else:
                unique_results[uid]['score'] = 10.0 if r['source_type'] == 'keyword' else r['score']
    
    sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
    return sorted_results[:10] # Lấy tối đa 10 kết quả

# --- 6. FALLBACK LOGIC ---
async def fallback_general_knowledge(question: str):
    system_prompt = f"""
    VAI TRÒ: Chuyên gia tư vấn du lịch và văn hóa Việt Nam.
    CÂU HỎI: "{question}"
    YÊU CẦU:
    1. Trả lời chính xác, khách quan, ngôn ngữ trang trọng (Professional Tone).
    2. TUYỆT ĐỐI KHÔNG sử dụng Emoji.
    3. Trình bày văn bản rõ ràng bằng Markdown.
    4. Tập trung vào thông tin thực tế.
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
    
    # --- 1. ROUTER THÔNG MINH (Intent + Location + Mode) ---
    router_prompt = f"""
    Phân tích câu hỏi: "{question}".
    Nhiệm vụ:
    1. Intent: "greeting" | "bus" | "tourism".
    2. Location: Trích xuất tên Tỉnh/Thành phố/Quận Huyện (nếu có).
    3. Mode: 
       - "list": Nếu hỏi danh sách, liệt kê, số lượng (VD: "Có những chùa nào", "Các địa điểm đẹp", "Liệt kê...").
       - "detail": Nếu hỏi chi tiết về một địa điểm cụ thể (VD: "Giới thiệu chùa Sùng Nghiêm", "Thuyết minh về...").
    
    JSON Output: {{ "intent": "...", "location": "...", "mode": "..." }}
    """
    
    intent = "tourism"
    start_loc = None
    end_loc = None
    location_filter = None
    mode = "detail"

    try:
        res = await asyncio.to_thread(llm_model.generate_content, router_prompt)
        clean = res.text.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(clean)
        intent = parsed.get("intent", "tourism")
        location_filter = parsed.get("location")
        mode = parsed.get("mode", "detail")
    except: pass

    # --- 2. LOGIC ĐIỀU HƯỚNG THỦ CÔNG (QUAN TRỌNG: ĐƯA LÊN TRƯỚC HẾT) ---
    lower_q = question.lower()
    
    # [FIX] Kiểm tra Intent BUS độc lập (không phụ thuộc vào câu hỏi nghi vấn)
    if "đến" in lower_q and ("từ" in lower_q or "đi" in lower_q or "tìm đường" in lower_q or "chỉ đường" in lower_q):
        intent = "bus"
    
    # Kiểm tra Intent GREETING hoặc ép về TOURISM nếu có từ để hỏi
    has_question_word = any(marker in lower_q for marker in QUESTION_MARKERS)
    
    if intent != "bus": # Chỉ check tiếp nếu chưa phải là bus
        if has_question_word:
            if intent == "greeting": intent = "tourism" # Ép về tourism nếu AI nhận nhầm greeting
        else:
            greeting_words = ["xin chào", "hello", "hi bot", "chào bạn", "alo", "hi there"]
            if any(w == lower_q or lower_q.startswith(w) for w in greeting_words) and len(lower_q) < 20:
                intent = "greeting"

    if location_filter and mode == "detail":
        if any(w in lower_q for w in ["nào", "gì", "những", "các", "đâu"]):
            mode = "list"

    logger.info(f"🔍 INTENT: {intent} | LOC: {location_filter} | MODE: {mode}")

    if intent == "greeting":
        greeting_msg = (
            "Kính chào Quý khách. Tôi là Trợ lý AI chuyên trách về Văn hóa, Du lịch và Giao thông công cộng.\n\n"
            "Tôi có thể hỗ trợ Quý khách:\n"
            "- Tra cứu lộ trình xe buýt tối ưu.\n"
            "- Cung cấp thông tin chuyên sâu về di tích, danh lam thắng cảnh Việt Nam.\n\n"
            "Quý khách cần tìm hiểu thông tin gì hôm nay?"
        )
        return {"answer": greeting_msg, "sources": [], "images": []}

    # --- XỬ LÝ BUS (CẬP NHẬT LOGIC BUTTON & PROMPT) ---
    if intent == "bus":
        # Parse điểm đi/đến thủ công nếu AI chưa bắt được
        if not start_loc or not end_loc:
            if "đến" in lower_q:
                try:
                    parts = lower_q.split("đến")
                    end_loc = parts[1].strip()
                    start_raw = parts[0]
                    for w in ["đi từ", "tìm đường từ", "chỉ đường từ", "từ", "đường đi"]:
                        start_raw = start_raw.replace(w, "")
                    start_loc = start_raw.strip()
                except: pass
        
        if start_loc and end_loc:
            try:
                # Gọi BusBot
                bus_result = await asyncio.to_thread(bus_bot.solve_route, start_loc, end_loc)
                
                # TRƯỜNG HỢP 1: CÓ NHIỀU LỰA CHỌN (AMBIGUOUS) -> TRẢ VỀ OPTIONS CHO CLIENT
                if bus_result.get("status") == "ambiguous":
                    return {
                        "answer": bus_result["message"], 
                        "options": bus_result["options"], # List các nút bấm
                        "context_type": "bus_ambiguity", # Đánh dấu để client biết
                        "original_request": {"start": start_loc, "end": end_loc, "type": bus_result["point_type"]},
                        "sources": [], 
                        "images": []
                    }

                # TRƯỜNG HỢP 2: LỖI
                if bus_result.get("status") == "error":
                    return {"answer": bus_result["message"], "sources": [], "images": []}
                
                # TRƯỜNG HỢP 3: THÀNH CÔNG (CÓ LỘ TRÌNH)
                raw_text = bus_result["text"]
                path_coords = bus_result.get("path_coords", [])
                
                google_link = ""
                if path_coords:
                    s = path_coords[0]
                    e = path_coords[-1]
                    google_link = f"https://www.google.com/maps/dir/?api=1&origin={s[1]},{s[0]}&destination={e[1]},{e[0]}&travelmode=transit"

                # Prompt này ép AI phải hiển thị đúng bảng giá vé 3 loại
                polish_prompt = f"""
                Dữ liệu lộ trình: \"\"\"{raw_text}\"\"\"
                YÊU CẦU: Viết lại hướng dẫn di chuyển chuyên nghiệp.
                1. Trình bày các bước rõ ràng.
                2. In đậm tên trạm, số xe.
                3. Bắt buộc hiển thị Bảng giá vé (Gồm: Vé thường, HSSV, Người cao tuổi) dựa trên dữ liệu.
                4. Văn phong lịch sự, không emoji.
                """
                final_res = await asyncio.to_thread(llm_model.generate_content, polish_prompt)
                
                answer_text = final_res.text
                if google_link:
                    answer_text += f"\n\n🔗 **[Xem bản đồ lộ trình trên Google Maps]({google_link})**"

                logger.info(f"DONE [BUS]: {time.time()-start_time:.2f}s")
                return {"answer": answer_text, "sources": [], "images": []}
            except Exception as e:
                return {"answer": f"Lỗi hệ thống: {str(e)}", "sources": [], "images": []}
        else:
            return {"answer": "Vui lòng cung cấp điểm đi và điểm đến (Ví dụ: Từ Bến Thành đến Đầm Sen) để tôi tìm lộ trình.", "sources": [], "images": []}

    # --- XỬ LÝ TOURISM (FULL LOGIC) ---
    final_province = request.province if request.province else location_filter
    search_results = await hybrid_search_tourism(question, final_province)
    
    if search_results:
        # 1. Xử lý ảnh: CHỈ LẤY URL ONLINE (Bỏ qua local path theo yêu cầu)
        image_urls = []
        for item in search_results:
            node_data = item.get('node', {})
            
            # Chỉ lấy trường image_url
            url = node_data.get('image_url')
            
            # Kiểm tra hợp lệ (phải bắt đầu bằng http)
            if url and isinstance(url, str) and url.startswith("http"):
                image_urls.append(url)
        
        # Lọc trùng và lấy tối đa 2 ảnh (THEO YÊU CẦU MỚI)
        final_imgs = list(dict.fromkeys(image_urls))[:2] 

        # 2. Tạo Context cho LLM
        context_str = "\n".join([f"- {item['node']['name']}: {item['node']['content']}" for item in search_results])
        
        # 3. Chọn Prompt theo Mode (List vs Detail)
        if mode == "list":
            rag_prompt = f"""
            VAI TRÒ: Hướng dẫn viên du lịch chuyên nghiệp.
            NHIỆM VỤ: Liệt kê danh sách các địa điểm.
            DỮ LIỆU CUNG CẤP: {context_str}
            CÂU HỎI NGƯỜI DÙNG: "{question}"
            
            YÊU CẦU TRẢ LỜI:
            1. Mở đầu: "Tại {location_filter if location_filter else 'khu vực này'}, có các địa điểm nổi tiếng sau:"
            2. Liệt kê danh sách (tối thiểu 3 điểm nếu có trong dữ liệu).
            3. Mỗi điểm mô tả ngắn gọn 2-3 câu về điểm đặc sắc nhất.
            4. Văn phong trang trọng, KHÔNG sử dụng emoji.
            """
        else:
            rag_prompt = f"""
            VAI TRÒ: Nhà nghiên cứu văn hóa và Hướng dẫn viên cao cấp.
            DỮ LIỆU CUNG CẤP: {context_str}
            CÂU HỎI NGƯỜI DÙNG: "{question}"
            
            YÊU CẦU TRẢ LỜI:
            1. Phong cách: Trang trọng, uyên bác, giàu cảm xúc. TUYỆT ĐỐI KHÔNG dùng emoji.
            2. Cấu trúc bài thuyết minh:
               - Mở đầu: Giới thiệu ấn tượng.
               - Thân bài: Chi tiết lịch sử, kiến trúc, giá trị văn hóa (Sử dụng gạch đầu dòng để trình bày).
               - Kết luận: Ý nghĩa của di tích.
            3. Chỉ sử dụng thông tin từ dữ liệu cung cấp, không bịa đặt.
            """

        res = await asyncio.to_thread(llm_model.generate_content, rag_prompt)
        logger.info(f"DONE [RAG]: {time.time()-start_time:.2f}s")
        return {"answer": res.text, "sources": search_results, "images": final_imgs}
    
    else:
        general_answer = await fallback_general_knowledge(question)
        logger.info(f"DONE [FALLBACK]: {time.time()-start_time:.2f}s")
        return {"answer": general_answer, "sources": [], "images": []}

# --- API XỬ LÝ ẢNH ĐẦU VÀO ---
@app.post("/chat_with_image")
async def chat_with_image_endpoint(file: UploadFile = File(...), question: str = Form(...)):
    start_time = time.time()
    logger.info(f"REQ [IMG]: {question}")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 1. Nhận diện địa điểm
        vision_prompt = "Đây là địa điểm nào? Chỉ trả về tên chính xác. Nếu không biết, trả về 'Unknown'."
        loop = asyncio.get_running_loop()
        vision_res = await loop.run_in_executor(None, lambda: llm_model.generate_content([vision_prompt, image]))
        detected_name = vision_res.text.strip()
        
        if "Unknown" in detected_name or len(detected_name) < 2:
            return {"detected_location": None, "answer": "Hệ thống chưa nhận diện được địa điểm trong ảnh. Vui lòng cung cấp ảnh rõ hơn.", "sources": [], "images": []}
             
        # 2. Tìm kiếm thông tin trong Neo4j
        search_results = await hybrid_search_tourism(detected_name)
        
        # 3. Lấy ảnh minh họa (Chỉ lấy URL Online)
        image_urls = []
        if search_results:
            for item in search_results:
                node_data = item.get('node', {})
                url = node_data.get('image_url')
                if url and isinstance(url, str) and url.startswith("http"): 
                    image_urls.append(url)
        
        # Lấy tối đa 2 ảnh
        final_imgs = list(dict.fromkeys(image_urls))[:2]

        # 4. Trả lời câu hỏi
        if search_results:
            context_str = "\n".join([f"- {item['node']['name']}: {item['node']['content']}" for item in search_results[:3]])
            final_prompt = f"""
            VAI TRÒ: Chuyên gia văn hóa.
            ĐỊA ĐIỂM NHẬN DIỆN TỪ ẢNH: {detected_name}
            DỮ LIỆU: {context_str}
            CÂU HỎI: {question}
            YÊU CẦU: Trả lời chuyên nghiệp, cấu trúc rõ ràng, KHÔNG emoji.
            """
            final_res = await loop.run_in_executor(None, lambda: llm_model.generate_content(final_prompt))
            return {"detected_location": detected_name, "answer": final_res.text, "sources": search_results, "images": final_imgs}
        else:
            general_prompt = f"Địa điểm trong ảnh là '{detected_name}'. Câu hỏi: '{question}'. Hãy trả lời chi tiết với văn phong chuyên gia, không dùng emoji."
            final_res = await loop.run_in_executor(None, lambda: llm_model.generate_content(general_prompt))
            return {"detected_location": detected_name, "answer": final_res.text, "sources": [], "images": []}

    except Exception as e:
        logger.error(f"ERROR: {e}")
        return {"answer": "Đã xảy ra lỗi trong quá trình xử lý ảnh.", "sources": [], "images": []}