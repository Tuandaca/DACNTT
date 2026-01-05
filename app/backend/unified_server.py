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

# --- [CRITICAL FIX] HÀM CHUẨN HÓA AN TOÀN ---
def normalize_text(text: Union[str, List[str], None]) -> str:
    """
    Chuẩn hóa text đầu vào, xử lý cả trường hợp text bị truyền vào là List
    để tránh lỗi AttributeError: 'list' object has no attribute 'replace'
    """
    if text is None: return ""
    if isinstance(text, list):
        # Nếu là list, lấy phần tử đầu tiên hoặc join lại tùy ngữ cảnh.
        # Ở đây ta lấy phần tử đầu tiên non-empty làm đại diện
        text = next((t for t in text if t), "")
    
    if not isinstance(text, str):
        text = str(text)

    return unicodedata.normalize('NFC', text).lower().strip()

# --- [FIXED V4] HÀM LÀM SẠCH CÓ ĐIỀU KIỆN ---
def clean_entity_name(raw_text: str, use_alias: bool = False) -> str:
    if not raw_text: return ""
    
    cleaned = raw_text.lower().strip()
    
    # 1. CẮT PREFIX (Giữ nguyên)
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
            
    # 2. CẮT SUFFIX (Giữ nguyên)
    suffixes = [
        " như thế nào", " thế nào", " ra sao", " làm sao", 
        " bằng cách nào", " đi đường nào",
        " được không", " có được không", " không ạ", " không",
        " số mấy", " bao nhiêu", " mất bao lâu",
        " ở đâu", " chỗ nào",
        " nhé", " nha", " nhỉ", " vậy", " ạ", " hả", " hử"
    ]
    
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()

    # 3. Double Check từ nối
    secondary_stopwords = ["từ ", "đến ", "về ", "tới ", "sang ", "qua ", "của "]
    for sw in secondary_stopwords:
        if cleaned.startswith(sw.strip()):
            cleaned = cleaned.replace(sw.strip(), "", 1).strip()

    # --- [CHỈ ÁP DỤNG KHI LÀ TOURISM] ---
    if use_alias:
        ENTITY_ALIASES = {
            "dinh độc lập": "hội trường thống nhất",
            "dinh norodom": "hội trường thống nhất",
            "phủ đầu rồng": "hội trường thống nhất",
            
            "nhà thờ đức bà": "vương cung thánh đường chính tòa đức bà sài gòn",
            "nhà thờ lớn sài gòn": "vương cung thánh đường chính tòa đức bà sài gòn",
            
            "bưu điện thành phố": "bưu điện trung tâm sài gòn",
            "bưu điện sài gòn": "bưu điện trung tâm sài gòn",
            
            "bến nhà rồng": "bảo tàng hồ chí minh",
            "bảo tàng bến nhà rồng": "bảo tàng hồ chí minh",
            
            "chợ lớn": "chợ bình tây",
            "landmart 81": "landmark 81",
            "lăng bác": "lăng chủ tịch hồ chí minh"
        }
        
        if cleaned in ENTITY_ALIASES:
            return ENTITY_ALIASES[cleaned].title()

    return cleaned.title()

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

# --- 5. LOGIC TÌM KIẾM TOURISM (ĐÃ FIX CRASH) ---
async def hybrid_search_tourism(text: str, province_filter: Union[str, List[str]] = None):
    # Luôn làm sạch input trước
    text = clean_entity_name(text)
    search_text_norm = normalize_text(text)
    records = []
    
    async with tourism_driver.session() as session:
        # Xử lý province_filter an toàn
        p_filter_cypher = ""
        if province_filter:
            # Dùng hàm normalize_text đã fix ở trên để xử lý cả list lẫn str
            p_filter_cypher = normalize_text(province_filter)

        # Query 1: Filter theo Tỉnh (Ưu tiên cao)
        if len(p_filter_cypher) > 2:
            cypher_loc = """
            MATCH (node:Searchable)-[:LOCATED_IN]->(p:Province)
            WHERE toLower(p.name) CONTAINS $province_norm
            AND (toLower(node.name) CONTAINS $text_norm OR toLower(node.content) CONTAINS $text_norm)
            RETURN elementId(node) as id, node, 1.5 as score, p.name as province_name, 'location_filter' as source_type
            LIMIT 15
            """
            try:
                r_loc = await session.run(cypher_loc, province_norm=p_filter_cypher, text_norm=search_text_norm)
                records.extend([record.data() async for record in r_loc])
            except Exception as e:
                logger.error(f"Error Loc Search: {e}")

        # Query 2: Vector & Keyword Search (Nếu ít kết quả)
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
            except Exception as e:
                logger.error(f"Vector Encode Error: {e}")

    # Rerank & Deduplicate
    unique_results = {}
    for r in records:
        uid = r['id']
        
        # Logic giảm điểm nếu sai tỉnh (Đã fix lỗi so sánh list)
        if province_filter and r.get('province_name'):
            r_province_norm = normalize_text(r['province_name'])
            is_match = False
            
            # Chuẩn hóa filter đầu vào thành list để dễ so sánh
            filter_list = province_filter if isinstance(province_filter, list) else [province_filter]
            
            for p in filter_list:
                if normalize_text(p) in r_province_norm:
                    is_match = True
                    break
            
            if not is_match:
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

# --- [CRITICAL FIX] PROMPT BẢO VỆ CHỐNG LEAK ---
async def fallback_general_knowledge(question: str):
    system_prompt = f"""
    VAI TRÒ: Bạn là Trợ lý ảo chuyên về Du lịch & Văn hóa Việt Nam.
    
    DANH SÁCH CẤM (TUYỆT ĐỐI KHÔNG TRẢ LỜI):
    1. LẬP TRÌNH: Code, Python, Java, C++, SQL, Docker, Fix bug, Lỗi phần mềm.
    2. TÀI CHÍNH: Lãi suất, Ngân hàng, Chứng khoán, Tiền ảo, Vay vốn, Đất nền.
    3. Y TẾ: Chữa bệnh, Thuốc, Bác sĩ, Triệu chứng, Đau nhức.
    4. GIAO THÔNG LIÊN TỈNH: Máy bay, Tàu hỏa (trừ khi hỏi ga tàu tại TP.HCM), Xe khách đi tỉnh.
       
    HÀNH ĐỘNG BẮT BUỘC KHI GẶP CHỦ ĐỀ CẤM:
    - Trả lời: "Xin lỗi, tôi chỉ là trợ lý du lịch và không có chuyên môn về lĩnh vực này. Vui lòng tham khảo ý kiến chuyên gia."
    - KHÔNG giải thích thêm, KHÔNG đưa ra lời khuyên "sơ sơ".
    - KHÔNG yêu cầu gửi ảnh.
    
    CÂU HỎI: "{question}"
    TRẢ LỜI NGẮN GỌN:
    """
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, lambda: llm_model.generate_content(system_prompt))
    return response.text

class ChatRequest(BaseModel):
    question: str
    province: Optional[Union[str, List[str]]] = None # Cho phép cả List và String

# ==========================================
#               MAIN CHAT API
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    raw_question = request.question.strip()
    question = re.sub(r'[\\/*.,?]+$', '', raw_question).strip()
    
    logger.info(f"REQ: {question}")
    lower_q = question.lower()
    
    # --- NHẬN DIỆN INTENT THÔNG MINH ---
    is_bus_intent = False
    
    # Danh sách từ khóa Du lịch/Ngoại tỉnh (Ưu tiên cao hơn Bus)
    tourism_keywords = [
        "lập kế hoạch", "kế hoạch", "lịch trình", "đi chơi", "du lịch", "tham quan", 
        "tour", "máy bay", "khách sạn", "nhà hàng", "ăn gì", "chơi gì", 
        "tàu hỏa", "tàu cao tốc", "xe khách", "limousine",
        "hà nội", "đà nẵng", "huế", "đà lạt", "phú quốc", "nha trang", "côn đảo", "quy nhơn", "sapa"
    ]
    
    is_tourism_override = any(k in lower_q for k in tourism_keywords)

    if not is_tourism_override:
        dest_markers = [" đến ", " tới ", " về ", " sang ", " qua ", " ra "] 
        strong_bus_words = ["xe bus", "xe buýt", "buýt", "tuyến xe", "trạm xe", "số mấy", "metro", "tàu điện"]
        
        if any(w in lower_q for w in strong_bus_words):
            is_bus_intent = True
        elif any(m.strip() in lower_q for m in dest_markers) and \
             any(w in lower_q for w in ["từ", "đi", "đường", "cách", "lộ trình", "chỉ đường"]):
            is_bus_intent = True

    # Prompt Router phân loại
    router_prompt = f"""
    Phân tích câu hỏi: "{question}".
    Nhiệm vụ:
    1. Intent: 
       - "bus": Chỉ hỏi về xe buýt/metro nội thành TP.HCM.
       - "tourism": Hỏi về du lịch, văn hóa, địa danh Việt Nam.
       - "greeting": Chào hỏi xã giao.
       - "ood": Tất cả chủ đề khác (Y tế, IT, Tài chính, Bất động sản, Giao thông liên tỉnh).
       
    2. Location: Trích xuất tên địa danh (NẾU CÓ).
    3. Mode: "list" | "detail".
    JSON Output: {{ "intent": "...", "location": "...", "mode": "..." }}
    """
    
    intent = "bus" if is_bus_intent else "tourism"
    location_filter = None
    mode = "detail"

    try:
        if not is_bus_intent: 
            res = await asyncio.to_thread(llm_model.generate_content, router_prompt)
            clean = res.text.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(clean)
            intent = parsed.get("intent", "tourism")
            location_filter = parsed.get("location")
            mode = parsed.get("mode", "detail")
    except: pass

    # Override nếu Router sai
    if intent == "bus" and is_tourism_override:
        intent = "tourism"
    
    has_question_word = any(marker in lower_q for marker in QUESTION_MARKERS)
    if intent != "bus":
        if has_question_word and intent == "greeting":
            intent = "tourism"
        elif not has_question_word:
            greeting_words = ["xin chào", "hello", "hi bot", "chào bạn", "alo"]
            if any(w == lower_q or lower_q.startswith(w) for w in greeting_words) and len(lower_q) < 20:
                intent = "greeting"

    # [FIX] Clean location từ Router luôn cho chắc
    if location_filter:
        location_filter = clean_entity_name(str(location_filter))

    logger.info(f"🔍 INTENT: {intent} | LOC: {location_filter}")

    if intent == "greeting":
        return {"answer": "Kính chào Quý khách. Tôi là Trợ lý AI chuyên trách về Văn hóa & Giao thông.\nTôi có thể giúp bạn tra cứu lộ trình xe buýt TP.HCM hoặc thông tin du lịch Việt Nam.", "sources": [], "images": []}

    # --- XỬ LÝ BUS (ĐÃ FIX LOGIC TRÍCH XUẤT) ---
    if intent == "bus":
        start_loc = None
        end_loc = None
        separators = [" đến ", " tới ", " về ", " sang ", " qua ", " ra "]
        # Mở rộng prefix để bắt dính input người dùng
        start_prefixes = [
            "đi từ", "từ", "tìm đường từ", "chỉ đường từ", "đường đi từ", 
            "lộ trình từ", "xe buýt từ", "bắt xe từ", "ghé", "chạy từ", "hướng dẫn bắt xe buýt từ",
            "lộ trình xe buýt đi từ", "lộ trình đi xe buýt từ"
        ]

        found_sep = None
        for sep in separators:
            if sep in lower_q:
                found_sep = sep
                break
        
        if found_sep:
            try:
                parts = lower_q.split(found_sep, 1)
                end_loc = parts[1].strip()
                start_raw = parts[0].strip()
                for prefix in start_prefixes:
                    if start_raw.startswith(prefix):
                        start_raw = start_raw[len(prefix):].strip()
                        break 
                start_loc = start_raw
            except: pass
        
        if start_loc and end_loc:
            try:
                # [CORE FIX] Gọi hàm làm sạch thần thánh
                start_loc = clean_entity_name(start_loc)
                end_loc = clean_entity_name(end_loc)
                
                logger.info(f"BUS SEARCH: Start='{start_loc}' | End='{end_loc}'")

                bus_result = await asyncio.to_thread(bus_bot.solve_route, start_loc, end_loc)
                
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
                3. Bắt buộc hiển thị Bảng giá vé.
                4. Văn phong lịch sự, không emoji.
                """
                final_res = await asyncio.to_thread(llm_model.generate_content, polish_prompt)
                
                answer_text = final_res.text
                if google_link:
                    answer_text += f"\n\n🔗 **[Xem bản đồ lộ trình trên Google Maps]({google_link})**"

                return {"answer": answer_text, "sources": [], "images": []}
            except Exception as e:
                logger.error(f"BUS ERROR: {e}")
                return {"answer": "Hệ thống đang gặp sự cố khi tra cứu xe buýt. Vui lòng thử lại sau.", "sources": [], "images": []}
        else:
            return {"answer": "Vui lòng cung cấp điểm đi và điểm đến (Ví dụ: Từ Bến Thành tới Aeon Mall) để tôi tìm lộ trình.", "sources": [], "images": []}

    # --- XỬ LÝ OOD / TOURISM ---
    if intent == "ood":
        general_answer = await fallback_general_knowledge(question)
        return {"answer": general_answer, "sources": [], "images": []}

    # Xử lý province filter an toàn
    final_province = request.province if request.province else location_filter
    search_query = question
    
    if final_province:
        if isinstance(final_province, list) and len(final_province) > 0:
            search_query = final_province[0] 
        elif isinstance(final_province, str):
            search_query = final_province

    # Làm sạch lần cuối
    search_query = clean_entity_name(search_query)

    try:
        search_results = await hybrid_search_tourism(search_query, final_province)
    except Exception as e:
        logger.error(f"SEARCH ERROR: {e}")
        search_results = []
    
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
            VAI TRÒ: Trợ lý du lịch. NHIỆM VỤ: Liệt kê.
            DỮ LIỆU: {context_str}
            CÂU HỎI: "{question}"
            YÊU CẦU: Liệt kê (tối thiểu 3). Ghi rõ Tên và Tỉnh/Thành. KHÔNG emoji.
            """
        else:
            rag_prompt = f"""
            VAI TRÒ: Chuyên gia văn hóa & du lịch.
            DỮ LIỆU: {context_str}
            CÂU HỎI: "{question}"
            YÊU CẦU: Ngắn gọn. Cấu trúc: 1. Tổng quan (Vị trí Tỉnh/Thành), 2. Gạch đầu dòng chi tiết.
            """

        res = await asyncio.to_thread(llm_model.generate_content, rag_prompt)
        return {"answer": res.text, "sources": search_results, "images": final_imgs}
    
    else:
        # Fallback cuối cùng nếu không tìm thấy gì
        general_answer = await fallback_general_knowledge(question)
        return {"answer": general_answer, "sources": [], "images": []}

# --- 5. API XỬ LÝ ẢNH ---
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
            return {"detected_location": None, "answer": "Chưa nhận diện được địa điểm trong ảnh.", "sources": [], "images": []}
             
        search_results = await hybrid_search_tourism(detected_name)
        
        image_urls = []
        if search_results:
            for item in search_results:
                url = item.get('node', {}).get('image_url')
                if url and isinstance(url, str) and url.startswith("http"): image_urls.append(url)
        final_imgs = list(dict.fromkeys(image_urls))[:2]

        if search_results:
            context_str = "\n".join([f"- {item['node']['name']}: {item['node']['content']}" for item in search_results[:3]])
            final_prompt = f"""
            VAI TRÒ: Chuyên gia văn hóa.
            ĐỊA ĐIỂM: {detected_name}
            DỮ LIỆU: {context_str}
            CÂU HỎI: {question}
            YÊU CẦU: Trả lời chuyên nghiệp. KHÔNG emoji.
            """
            final_res = await loop.run_in_executor(None, lambda: llm_model.generate_content(final_prompt))
            return {"detected_location": detected_name, "answer": final_res.text, "sources": search_results, "images": final_imgs}
        else:
            general_prompt = f"Địa điểm: '{detected_name}'. Câu hỏi: '{question}'. Trả lời chi tiết. KHÔNG emoji."
            final_res = await loop.run_in_executor(None, lambda: llm_model.generate_content(general_prompt))
            return {"detected_location": detected_name, "answer": final_res.text, "sources": [], "images": []}

    except Exception as e:
        logger.error(f"IMAGE ERROR: {e}")
        return {"answer": "Đã xảy ra lỗi khi xử lý hình ảnh.", "sources": [], "images": []}