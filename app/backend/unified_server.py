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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

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
if os.path.exists("images_items"):
    app.mount("/images_items", StaticFiles(directory="images_items"), name="images")

# --- BLACKLIST GREETING ---
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
    "thông tin", "chi tiết", "giới thiệu", "kể về", "nói về"
]

# --- SEARCH LOGIC ---
async def hybrid_search_tourism(text: str, province_filter: str = None):
    search_text_norm = normalize_text(text)
    loop = asyncio.get_running_loop()
    vector = await loop.run_in_executor(None, lambda: model.encode(text).tolist())
    
    cypher_vector = "CALL db.index.vector.queryNodes('searchable_index', 50, $vector) YIELD node, score OPTIONAL MATCH (node)-[:LOCATED_IN]->(p:Province) RETURN elementId(node) as id, node, score, p.name as province_name, 'vector' as source_type"
    cypher_keyword = "MATCH (node:Searchable) WHERE toLower(node.name) CONTAINS $text_norm OPTIONAL MATCH (node)-[:LOCATED_IN]->(p:Province) RETURN elementId(node) as id, node, 1.0 as score, p.name as province_name, 'keyword' as source_type LIMIT 20"
    
    records = []
    async with tourism_driver.session() as session:
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
        if r['source_type'] == 'vector' and r['score'] < 0.70: continue
        if uid not in unique_results:
            unique_results[uid] = r
            unique_results[uid]['score'] = 10.0 if r['source_type'] == 'keyword' else r['score']
    
    return list(unique_results.values())[:5]

async def fallback_general_knowledge(question: str):
    system_prompt = f"""
    VAI TRÒ: Chuyên gia tư vấn du lịch và văn hóa Việt Nam.
    CÂU HỎI: "{question}"
    YÊU CẦU:
    1. Trả lời chính xác, khách quan, ngôn ngữ trang trọng (Professional Tone).
    2. KHÔNG sử dụng Emoji.
    3. Trình bày văn bản rõ ràng bằng Markdown.
    4. Tập trung vào thông tin thực tế.
    """
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, lambda: llm_model.generate_content(system_prompt))
    return response.text

class ChatRequest(BaseModel):
    question: str
    province: Optional[str] = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    question = request.question.strip()
    logger.info(f"REQ: {question}")
    
    # 1. ROUTER
    router_prompt = f"""
    Phân tích câu nói: "{question}".
    Hãy xác định Intent (Ý định) chính xác nhất:
    - "greeting": CHỈ LÀ câu chào hỏi xã giao thuần túy (Hello, Hi, Xin chào...). 
      LƯU Ý: Nếu câu chứa bất kỳ nội dung hỏi đáp nào -> BẮT BUỘC chọn "tourism" hoặc "bus".
    - "bus": Hỏi đường đi, xe buýt.
    - "tourism": Hỏi thông tin, kiến thức, địa điểm.
    Trả về JSON: {{ "intent": "greeting" | "bus" | "tourism", "start_point": "...", "end_point": "..." }}
    """
    intent = "tourism"
    start_loc = None
    end_loc = None

    try:
        res = await asyncio.to_thread(llm_model.generate_content, router_prompt)
        clean = res.text.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(clean)
        intent = parsed.get("intent", "tourism")
        start_loc = parsed.get("start_point")
        end_loc = parsed.get("end_point")
    except: pass

    # KILL SWITCH CHO GREETING
    lower_q = question.lower()
    has_question_word = any(marker in lower_q for marker in QUESTION_MARKERS)
    if has_question_word:
        if intent == "greeting": intent = "tourism"
        if "đến" in lower_q and ("từ" in lower_q or "đi" in lower_q): intent = "bus"
    else:
        greeting_words = ["xin chào", "hello", "hi bot", "chào bạn", "alo", "hi there", "chào ad"]
        if any(w == lower_q or lower_q.startswith(w) for w in greeting_words) and len(lower_q) < 20:
            intent = "greeting"

    logger.info(f"🔍 [INTENT]: {intent}")

    if intent == "greeting":
        greeting_msg = (
            "Xin chào! Tôi là Trợ lý AI chuyên về Giao thông & Văn hóa Việt Nam.\n\n"
            "Tôi có thể giúp bạn:\n"
            "- Tra cứu lộ trình xe buýt TP.HCM.\n"
            "- Tìm hiểu văn hóa, lịch sử, ẩm thực và du lịch.\n\n"
            "Bạn đang quan tâm đến vấn đề gì?"
        )
        logger.info(f"DONE [GREETING]: {time.time()-start_time:.2f}s")
        return {"answer": greeting_msg, "sources": [], "images": []}

    if intent == "bus" and (not start_loc or not end_loc):
        if "đến" in lower_q:
            try:
                parts = lower_q.split("đến")
                start_raw = parts[0]
                for w in ["đi từ", "tìm đường từ", "từ", "đường đi"]:
                    start_raw = start_raw.replace(w, "")
                start_loc = start_raw.strip()
                end_loc = parts[1].strip()
            except: pass

    # 2. XỬ LÝ BUS
    if intent == "bus":
        if start_loc and end_loc:
            try:
                bus_result = await asyncio.to_thread(bus_bot.solve_route, start_loc, end_loc)
                if bus_result.get("status") == "error":
                    return {"answer": bus_result["message"], "sources": [], "images": []}
                
                raw_text = bus_result["text"]
                path_coords = bus_result.get("path_coords", [])
                
                # --- TẠO LINK GOOGLE MAPS ---
                google_link = ""
                if path_coords:
                    # Coords từ Neo4j trả về dạng [lng, lat]
                    # Google Maps cần: origin=lat,lng & destination=lat,lng
                    s = path_coords[0]  # Điểm đầu [lng, lat]
                    e = path_coords[-1] # Điểm cuối [lng, lat]
                    google_link = f"https://www.google.com/maps/dir/?api=1&origin={s[1]},{s[0]}&destination={e[1]},{e[0]}&travelmode=transit"

                polish_prompt = f"""
                Dữ liệu lộ trình: \"\"\"{raw_text}\"\"\"
                YÊU CẦU: Viết lại hướng dẫn di chuyển chuyên nghiệp, rõ ràng.
                - Dùng danh sách số (1., 2., 3.).
                - In đậm tên trạm, số xe.
                - KHÔNG dùng emoji.
                - HIỂN THỊ ĐẦY ĐỦ GIÁ VÉ (Từng chặng và Tổng cộng).
                """
                final_res = await asyncio.to_thread(llm_model.generate_content, polish_prompt)
                
                answer_text = final_res.text
                if google_link:
                    answer_text += f"\n\n🔗 **[Bấm vào đây để xem bản đồ trên Google Maps]({google_link})**"

                logger.info(f"DONE [BUS]: {time.time()-start_time:.2f}s")
                # LƯU Ý: Không trả về path_coords để Client không vẽ map
                return {"answer": answer_text, "sources": [], "images": []}
            except Exception as e:
                return {"answer": f"Lỗi hệ thống: {str(e)}", "sources": [], "images": []}
        else:
            return {"answer": "Vui lòng cung cấp điểm đi và điểm đến để tôi tìm lộ trình xe buýt.", "sources": [], "images": []}

    # 3. XỬ LÝ TOURISM
    search_results = await hybrid_search_tourism(question, request.province)
    if search_results:
        image_urls = []
        for item in search_results[:3]:
            node_data = item.get('node', {})
            url = node_data.get('image_url') or node_data.get('image')
            if url and isinstance(url, str) and url.startswith("http"):
                image_urls.append(url)
        seen = set()
        uniq_imgs = [x for x in image_urls if not (x in seen or seen.add(x))]

        context_str = "\n".join([f"- {item['node']['name']}: {item['node']['content']}" for item in search_results])
        rag_prompt = f"""
        Bạn là chuyên gia văn hóa/du lịch. DỮ LIỆU: {context_str}. CÂU HỎI: "{question}"
        YÊU CẦU: Trả lời chuyên nghiệp, khách quan, Markdown. KHÔNG dùng emoji.
        """
        res = await asyncio.to_thread(llm_model.generate_content, rag_prompt)
        logger.info(f"DONE [RAG]: {time.time()-start_time:.2f}s")
        return {"answer": res.text, "sources": search_results, "images": uniq_imgs}
    else:
        general_answer = await fallback_general_knowledge(question)
        logger.info(f"DONE [FALLBACK]: {time.time()-start_time:.2f}s")
        return {"answer": general_answer, "sources": [], "images": []}

@app.post("/chat_with_image")
async def chat_with_image_endpoint(file: UploadFile = File(...), question: str = Form(...)):
    start_time = time.time()
    logger.info(f"REQ [IMG]: {question}")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        vision_prompt = "Đây là địa điểm nào? Chỉ trả về tên chính xác. Nếu không biết, trả về 'Unknown'."
        loop = asyncio.get_running_loop()
        vision_res = await loop.run_in_executor(None, lambda: llm_model.generate_content([vision_prompt, image]))
        detected_name = vision_res.text.strip()
        
        if "Unknown" in detected_name or len(detected_name) < 2:
             return {"detected_location": None, "answer": "Hệ thống chưa nhận diện được địa điểm trong ảnh.", "sources": [], "images": []}
             
        search_results = await hybrid_search_tourism(detected_name)
        image_urls = []
        if search_results:
             for item in search_results[:1]:
                node_data = item.get('node', {})
                url = node_data.get('image_url') or node_data.get('image')
                if url and isinstance(url, str) and url.startswith("http"): image_urls.append(url)

        if search_results:
            context_str = "\n".join([f"- {item['name']}: {item['content']}" for item in search_results[:3]])
            final_prompt = f"Địa điểm: {detected_name}. Dữ liệu: {context_str}. Câu hỏi: {question}. Yêu cầu trả lời chuyên nghiệp, không emoji."
            final_res = await loop.run_in_executor(None, lambda: llm_model.generate_content(final_prompt))
            return {"detected_location": detected_name, "answer": final_res.text, "sources": search_results, "images": image_urls}
        else:
            general_prompt = f"Địa điểm trong ảnh là '{detected_name}'. Câu hỏi: '{question}'. Hãy trả lời chi tiết với văn phong chuyên gia, không dùng emoji."
            final_res = await loop.run_in_executor(None, lambda: llm_model.generate_content(general_prompt))
            return {"detected_location": detected_name, "answer": final_res.text, "sources": [], "images": []}

    except Exception as e:
        logger.error(f"ERROR: {e}")
        return {"answer": "Đã xảy ra lỗi xử lý ảnh.", "sources": [], "images": []}