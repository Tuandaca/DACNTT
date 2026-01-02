import streamlit as st
import requests
import uuid
# ĐÃ BỎ IMPORT FOLIUM

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Hệ thống Hỗ trợ Du lịch & Giao thông TP.HCM",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CSS STYLE ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    header {visibility: hidden;} footer {visibility: hidden;}
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #e9ecef; }
    section[data-testid="stSidebar"] * { color: #212529 !important; }
    .stButton > button { background-color: #ffffff !important; color: #495057 !important; border: 1px solid #ced4da !important; border-radius: 6px; font-weight: 500; box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition: all 0.2s; width: 100%; }
    .stButton > button:hover { border-color: #0056b3 !important; color: #0056b3 !important; background-color: #e7f1ff !important; }
    .streamlit-expanderHeader { background-color: #ffffff !important; color: #0056b3 !important; border: 1px solid #dee2e6; border-radius: 6px; font-weight: 600 !important; }
    .streamlit-expanderContent { background-color: #f8f9fa !important; border: none !important; color: #212529 !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #ffffff !important; border: 1px dashed #adb5bd !important; color: #212529 !important; border-radius: 6px; }
    [data-testid="stFileUploaderDropzone"] div { color: #495057 !important; }
    [data-testid="stFileUploaderDropzone"] button { background-color: #e9ecef !important; color: #212529 !important; border: none !important; }
    div[data-testid="stChatMessageUser"] { background-color: #e7f1ff; color: #212529; border-radius: 10px; padding: 1rem; }
    div[data-testid="stChatMessageAssistant"] { background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 10px; padding: 1rem; color: #212529; }
</style>
""", unsafe_allow_html=True)

if "all_chats" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.all_chats = {new_id: {"title": "Phiên làm việc mới", "messages": []}}
    st.session_state.active_chat_id = new_id

def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.all_chats[new_id] = {"title": "Phiên làm việc mới", "messages": []}
    st.session_state.active_chat_id = new_id

def switch_chat(chat_id): st.session_state.active_chat_id = chat_id
def delete_current_chat():
    del st.session_state.all_chats[st.session_state.active_chat_id]
    if not st.session_state.all_chats: create_new_chat()
    else: st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0]

with st.sidebar:
    st.markdown("### 🗂️ Menu Chức năng")
    if st.button("➕ Tạo hội thoại mới", use_container_width=True, key="btn_new_chat"):
        create_new_chat()
        st.rerun()
    st.write("---")
    st.markdown("#### 🕒 Lịch sử tra cứu")
    for chat_id, chat_data in reversed(list(st.session_state.all_chats.items())):
        label = chat_data["title"][:22] + "..." if len(chat_data["title"]) > 25 else chat_data["title"]
        btn_type = "primary" if chat_id == st.session_state.active_chat_id else "secondary"
        if st.button(label, key=f"hist_{chat_id}", type=btn_type, use_container_width=True):
            switch_chat(chat_id)
            st.rerun()
    st.write("---")
    with st.expander("📷 Công cụ hình ảnh", expanded=True):
        uploaded_file = st.file_uploader("Tải ảnh lên để tra cứu", type=['jpg', 'png', 'jpeg'], key="img_uploader")
        if st.button("🗑️ Xóa dữ liệu chat", use_container_width=True, key="btn_del"):
            delete_current_chat()
            st.rerun()

# --- CHAT UI ---
current_chat = st.session_state.all_chats[st.session_state.active_chat_id]
messages = current_chat["messages"]

st.markdown("<h2 style='text-align: center; color: #0056b3; font-weight: 700;'>Ứng dụng hỏi đáp nhanh</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d;'>Trợ lý ảo hỗ trợ lộ trình xe buýt tại TP.HCM và thông tin văn hóa - du lịch Việt Nam </p>", unsafe_allow_html=True)
st.divider()

if not messages:
    st.info("👋 Hệ thống đã sẵn sàng. Vui lòng nhập câu hỏi hoặc tải ảnh lên để bắt đầu.")

for msg in messages:
    with st.chat_message(msg["role"]):
        if "image_data" in msg and msg["image_data"]:
            st.image(msg["image_data"], width=250)
        st.markdown(msg["content"])
        
        # Chỉ hiển thị ảnh du lịch, KHÔNG CÒN BẢN ĐỒ MAP Ở ĐÂY
        if "images" in msg and msg["images"]:
            st.markdown("**Hình ảnh tham khảo:**")
            cols = st.columns(len(msg["images"]))
            for i, img_url in enumerate(msg["images"]):
                with cols[i]:
                    st.image(img_url, use_column_width=True)

if prompt := st.chat_input("Nhập nội dung cần hỗ trợ..."):
    if len(messages) == 0: 
        st.session_state.all_chats[st.session_state.active_chat_id]["title"] = prompt
    user_msg = {"role": "user", "content": prompt}
    if uploaded_file: user_msg["image_data"] = uploaded_file.getvalue()
    st.session_state.all_chats[st.session_state.active_chat_id]["messages"].append(user_msg)
    
    with st.chat_message("user"):
        if uploaded_file: st.image(uploaded_file.getvalue(), width=250)
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Đang xử lý..."):
            try:
                api_url = "http://127.0.0.1:8000/chat_with_image" if uploaded_file else "http://127.0.0.1:8000/chat"
                if uploaded_file:
                    uploaded_file.seek(0)
                    files = {"file": uploaded_file}
                    data = {"question": prompt}
                    response = requests.post(api_url, files=files, data=data, timeout=60)
                else:
                    payload = {"question": prompt}
                    response = requests.post(api_url, json=payload, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    ans = data.get("answer", "Không có phản hồi.")
                    placeholder.markdown(ans)
                    assistant_msg = {"role": "assistant", "content": ans}
                    
                    if "images" in data and data["images"]:
                        assistant_msg["images"] = data["images"]
                        st.markdown("**Hình ảnh tham khảo:**")
                        cols = st.columns(len(data["images"]))
                        for i, img_url in enumerate(data["images"]):
                            with cols[i]:
                                st.image(img_url, use_column_width=True)
                                
                    st.session_state.all_chats[st.session_state.active_chat_id]["messages"].append(assistant_msg)
                    if uploaded_file: st.rerun()
                else:
                    placeholder.error(f"Lỗi: {response.status_code}")
            except Exception as e:
                placeholder.error(f"Lỗi: {e}")