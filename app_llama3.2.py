import streamlit as st
import os
import uuid
import atexit
import signal
import gc
import base64
import time
import tempfile
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# from crewai import Agent, Crew, Process, Task, LLM
# from crewai.tasks.task_output import TaskOutput
try:
    from src.agentic_rag.tools.serper_tool import SerperDevTool
except ImportError:
    SerperDevTool = None
    st.warning("SerperDevTool not available. Please check serper_tool.py for web search.")
from src.agentic_rag.tools.custom_tool import DocumentSearchTool
from src.agentic_rag.crew import build_langgraph_workflow
try:
    from src.agentic_rag.tools.chat_history import ChatHistoryStore
except Exception:
    ChatHistoryStore = None
import pytesseract

# Premium Deep Modern CSS & Layout
# Load Logo
import base64
logo_base64 = base64.b64encode(open("assets/Typhoon2.png", "rb").read()).decode()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

html, body, [class^='css'] {{
    font-family: 'Inter', 'Prompt', sans-serif !important;
}}

.stApp {{
    background: linear-gradient(120deg, #181c2f 0%, #232946 100%);
    color: #f3f6fa;
    min-height: 100vh;
}}

/* 🔹 Sidebar Styling */
section[data-testid="stSidebar"] {{
    width: 340px !important;
    background: rgba(36, 40, 59, 0.75);
    border-radius: 22px;
    padding: 40px 28px 32px 28px;
    margin: 32px 18px;
    box-shadow: 0 0 18px rgba(142, 148, 251, 0.3);
    backdrop-filter: blur(16px);
    color: #f3f6fa;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}}

/* 🔹 File Uploader */
.stFileUploader {{
    background: #1d1f33;
    border: 1.5px dashed #8f94fb;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    color: #ccc;
    font-weight: 500;
}}

.stFileUploader button {{
    background: linear-gradient(90deg, #4e54c8, #8f94fb);
    color: white;
    font-weight: bold;
    padding: 10px 22px;
    border-radius: 12px;
    border: none;
    margin-top: 12px;
}}

.stFileUploader button:hover {{
    background: linear-gradient(90deg, #8f94fb, #4e54c8);
    transform: scale(1.03);
    transition: all 0.2s ease-in-out;
}}

/* 🔹 Reset Chat Button */
.stButton > button {{
    background: linear-gradient(90deg, #4e54c8, #8f94fb);
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 14px;
    border: none;
    margin-top: 20px;
}}

.stButton > button:hover {{
    background: linear-gradient(90deg, #8f94fb, #4e54c8);
    transform: scale(1.03);
    box-shadow: 0 4px 14px rgba(78, 84, 200, 0.3);
}}

/* 🔹 Chat Messages */
.stChatMessage {{
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px 24px;
    margin: 12px 0;
    border: 1px solid rgba(255,255,255,0.08);
    color: #f3f6fa;
}}

.stChatMessage[data-testid="user"] {{
    background: linear-gradient(90deg, #4e54c8, #8f94fb);
    color: white;
    font-weight: 600;
    box-shadow: 0 6px 18px rgba(78, 84, 200, 0.2);
}}

.stChatMessage[data-testid="assistant"] {{
    background: rgba(36, 40, 59, 0.85);
    border: 1px solid rgba(255,255,255,0.06);
    color: #f3f6fa;
}}

/* 🔹 Input Bar Floating Bottom */
.stChatInputContainer {{
    position: fixed;
    left: 0; right: 0; bottom: 0;
    background: rgba(36, 40, 59, 0.92);
    padding: 16px 0;
    box-shadow: 0 -2px 16px rgba(31, 38, 135, 0.18);
    z-index: 100;
    display: flex;
    justify-content: center;
}}

.stTextInput {{
    width: 700px !important;
    max-width: 90vw;
}}

/* 🔹 Scrollbar */
::-webkit-scrollbar {{
    width: 10px;
    background: #232526;
}}

::-webkit-scrollbar-thumb {{
    background: #4e54c8;
    border-radius: 8px;
}}

/* 🔹 Main Logo and Title */
.main-header {{
    text-align: center;
    margin: 50px 0 30px 0;
}}

.main-logo {{
    width: 100px;
    margin-bottom: 16px;
    filter: drop-shadow(0 0 16px rgba(142, 148, 251, 0.5));
}}

.main-title {{
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 1.2px;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}}

/* 🔹 Expander Styling for Alternative Answers */
.streamlit-expanderHeader {{
    background: linear-gradient(90deg, #4e54c8, #8f94fb) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    border: none !important;
    margin: 8px 0 !important;
    transition: all 0.3s ease !important;
}}

.streamlit-expanderHeader:hover {{
    background: linear-gradient(90deg, #8f94fb, #4e54c8) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(78, 84, 200, 0.3) !important;
}}

.streamlit-expanderContent {{
    background: rgba(36, 40, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-top: 8px !important;
    backdrop-filter: blur(10px) !important;
}}

/* 🔹 Alternative Answer Styling */
.alternative-answer {{
    background: rgba(255, 255, 255, 0.03);
    border-left: 3px solid #8f94fb;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 8px;
    border-top-left-radius: 0;
    border-bottom-left-radius: 0;
}}
</style>

<!-- 🔹 Injected Logo + Title -->
<div class="main-header">
    <img class="main-logo" src="data:image/png;base64,{logo_base64}" />
    <div class="main-title">PDPA Assistant</div>
</div>
""", unsafe_allow_html=True)


# Remove unused log_task_output function (was for CrewAI)
# def log_task_output(task_output: TaskOutput):
#     """Callback function to log the output of each task in a structured way."""
#     agent_name = task_output.agent
#     task_description = task_output.description
#     output = task_output.raw
    
#     print("\n" + "="*50)
#     print(f"✅ Task Completed by: {agent_name}")
#     print(f"📝 Task: {task_description}")
    
#     # Specifically highlight the multi-answer and ranking outputs
#     if "Multiple Answer Generator" in agent_name:
#         print(f"\n🧠 Generated Candidate Answers:\n{output}")
#     elif "Candidate Answer Ranker" in agent_name:
#         print(f"\n⚖️ Ranked Answers:\n{output}")
#     else:
#         # Just print the raw output for other agents, as the verbose log is already detailed
#         print(f"Completed with output.")
        
#     print("="*50 + "\n")


# @st.cache_resource
# def load_llm():
#     llm = LLM(
#         model="ollama/hf.co/Float16-cloud/typhoon2-qwen2.5-7b-instruct-gguf:Q8_0",
#         base_url="http://localhost:11434"
#     )
#     return llm

# ===========================
#   Helper Functions
# ===========================
def is_pdpa_related(document_tool):
    """
    Checks if the uploaded file is related to PDPA by searching for PDPA-related terms in the document.
    
    Args:
        document_tool: The DocumentSearchTool instance initialized with the file
        
    Returns:
        bool: True if the file is likely PDPA-related, False otherwise
    """
    # PDPA-related keywords to check for
    pdpa_keywords = [
        "PDPA", "Personal Data Protection Act", "คุ้มครองข้อมูลส่วนบุคคล", "พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล", 
        "ข้อมูลส่วนบุคคล", "data controller", "data processor", "ผู้ควบคุมข้อมูล", "ผู้ประมวลผลข้อมูล",
        "สิทธิเจ้าของข้อมูล", "การประมวลผลข้อมูล", "การเก็บรวบรวมข้อมูล", "ฐานทางกฎหมาย"
    ]
    
    # Check if the document contains any PDPA-related keywords
    if hasattr(document_tool, 'raw_text') and document_tool.raw_text:
        text = document_tool.raw_text.lower()
        # Check for presence of keywords
        for keyword in pdpa_keywords:
            if keyword.lower() in text:
                return True
    
    return False

# ===========================
#   Define Agents & Tasks
# ===========================
def create_agents_and_tasks(pdf_tool, use_knowledge_base=True, file_query_mode=False):
    """สร้าง LangGraph workflow ที่ประกอบด้วย agent สำหรับค้นคว้าและสังเคราะห์คำตอบเกี่ยวกับ PDPA ให้มีคุณภาพสูง
    โดยใช้เครื่องมือค้นหา PDF (pdf_tool) และเครื่องมือค้นหาบนเว็บ"""
    
    # ตรวจสอบ SerperDevTool และ API key
    if SerperDevTool:
        try:
            # ตรวจสอบว่ามี SERPER_API_KEY หรือไม่
            if os.getenv("SERPER_API_KEY"):
                print("✅ SerperDevTool API key found. Web search will be enabled.")
            else:
                print("⚠️ SERPER_API_KEY not found. Web search will be disabled.")
                st.info("🌐 Web search is available but requires SERPER_API_KEY. Add it to your .env file to enable web search.")
        except Exception as e:
            print(f"❌ Error checking SerperDevTool: {e}")
    
    # สร้าง LangGraph workflow
    workflow = build_langgraph_workflow(pdf_tool=pdf_tool, use_knowledge_base=use_knowledge_base)
    return workflow

# ===========================
#   Streamlit State Setup
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    # Mirror to module-level for cleanup without Streamlit context
    try:
        CHAT_SESSION_ID = st.session_state.session_id
    except Exception:
        CHAT_SESSION_ID = None

if "chat_store" not in st.session_state:
    try:
        if ChatHistoryStore is not None:
            qdrant_url = os.getenv("QDRANT_URL2")
            qdrant_api_key = os.getenv("QDRANT_API_KEY2")
            if qdrant_url:
                st.session_state.chat_store = ChatHistoryStore(
                    collection_name="rag_chat_history",
                    qdrant_url=qdrant_url,
                    qdrant_api_key=qdrant_api_key,
                )
                # Mirror to module-level for cleanup without Streamlit context
                try:
                    CHAT_STORE_REF = st.session_state.chat_store
                except Exception:
                    CHAT_STORE_REF = None
                # Load existing messages for this session
                st.session_state.messages = st.session_state.chat_store.list_messages(st.session_state.session_id)
            else:
                st.session_state.chat_store = None
                st.info("Chat history disabled. Set QDRANT_URL2 to enable Qdrant-backed history.")
        else:
            st.session_state.chat_store = None
    except Exception as e:
        st.warning(f"Chat history store unavailable: {e}")
        st.session_state.chat_store = None

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None

if "knowledge_base_tool" not in st.session_state:
    knowledge_files = os.path.join("knowledge")
    if os.path.exists(knowledge_files) and os.listdir(knowledge_files):
        try:
            st.session_state.knowledge_base_tool = DocumentSearchTool(file_path=knowledge_files)
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")
            st.session_state.knowledge_base_tool = None
    else:
        st.session_state.knowledge_base_tool = None

if "langgraph_workflow" not in st.session_state:
    st.session_state.langgraph_workflow = build_langgraph_workflow()

if "using_uploaded_file" not in st.session_state:
    st.session_state.using_uploaded_file = False

if "is_pdpa_related" not in st.session_state:
    st.session_state.is_pdpa_related = False

# ===========================
#   Helper Functions
# ===========================
def build_conversation_context(messages, max_turns=3):
    """รวมประวัติการสนทนาล่าสุดเพื่อให้บริบทเพิ่มเติม"""
    if not messages:
        return ""
    
    # จำกัดจำนวนรอบการสนทนาเพื่อไม่ให้ context ยาวเกินไป
    start_idx = max(0, len(messages) - (max_turns * 2))
    recent_messages = messages[start_idx:]
    
    context = []
    for msg in recent_messages:
        role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
        context.append(f"{role_prefix}{msg['content']}")
    
    return "\n".join(context)

def reset_chat():
    """ล้างประวัติการสนทนา"""
    try:
        if st.session_state.get("chat_store") and st.session_state.get("session_id"):
            st.session_state.chat_store.reset_session(st.session_state.session_id)
    except Exception as e:
        st.warning(f"ไม่สามารถล้างแชตบน Qdrant ได้: {e}")
    st.session_state.messages = []
    perform_periodic_gc()


# ===========================
#   Cleanup on Exit
# ===========================
def _cleanup_on_exit():
    # Use module-level references to avoid Streamlit ScriptRunContext at exit
    try:
        mode = os.getenv("CHAT_HISTORY_CLEANUP_MODE", "session").lower()
        chat_store = globals().get("CHAT_STORE_REF")
        session_id = globals().get("CHAT_SESSION_ID")
        if chat_store:
            if mode == "collection":
                chat_store.drop_collection()
            elif session_id:
                chat_store.reset_session(session_id)
    except Exception:
        pass


def _signal_handler(signum, frame):
    _cleanup_on_exit()
    raise SystemExit(0)


atexit.register(_cleanup_on_exit)
try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    # Some environments may not allow signal handling
    pass

def perform_periodic_gc():
    """ทำ garbage collection เพื่อลดการใช้หน่วยความจำ"""
    try:
        if st.session_state.pdf_tool and hasattr(st.session_state.pdf_tool, "_perform_gc"):
            st.session_state.pdf_tool._perform_gc()
        gc.collect()
    except Exception as e:
        st.error(f"Error during garbage collection: {str(e)}")

def display_pdf(file_bytes: bytes, file_name: str):
    """แสดงไฟล์ PDF ใน iframe"""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <div style="display: flex; justify-content: center; margin: 20px 0;">
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="600px" 
            style="border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; box-shadow: 0 4px 30px rgba(0,0,0,0.1);"
            type="application/pdf"
        >
        </iframe>
    </div>
    """
    st.markdown(f"<h3 style='text-align: center; margin-bottom: 16px; color: #fff;'>รายละเอียดเอกสาร: {file_name}</h3>", unsafe_allow_html=True)
    st.markdown(pdf_display, unsafe_allow_html=True)

# ===========================
#   Sidebar
# ===========================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>อัปโหลดเอกสาร PDPA</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("เลือกไฟล์ PDF", type=["pdf"])

    if uploaded_file is not None:
        # ถ้ามีไฟล์ใหม่ถูกอัปโหลด ให้สร้าง PDF tool ชั่วคราว
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.spinner("กำลังประมวลผลเอกสาร... โปรดรอสักครู่..."):
                # คืนทรัพยากรจากไฟล์เก่า (ถ้ามี)
                if st.session_state.pdf_tool is not None:
                    if hasattr(st.session_state.pdf_tool, "release_resources"):
                        st.session_state.pdf_tool.release_resources()
                
                # สร้าง tool ใหม่สำหรับไฟล์ที่อัปโหลด
                try:
                    st.session_state.pdf_tool = DocumentSearchTool(file_path=temp_file_path)
                    
                    # ตรวจสอบว่าไฟล์เกี่ยวข้องกับ PDPA หรือไม่
                    st.session_state.is_pdpa_related = is_pdpa_related(st.session_state.pdf_tool)
                    
                    # สร้าง workflow ใหม่สำหรับไฟล์ที่อัปโหลด
                    st.session_state.langgraph_workflow = create_agents_and_tasks(
                        st.session_state.pdf_tool, 
                        use_knowledge_base=False,
                        file_query_mode=True
                    )
                    st.session_state.using_uploaded_file = True
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์: {str(e)}")
                    st.session_state.pdf_tool = None
                    st.session_state.is_pdpa_related = False
            
        if st.session_state.pdf_tool:
            if st.session_state.is_pdpa_related:
                st.success("ประมวลผลเอกสารเสร็จสิ้น! เอกสารนี้เกี่ยวข้องกับ PDPA คุณสามารถถามคำถามเกี่ยวกับเอกสารนี้ได้เลย")
            else:
                st.warning("ประมวลผลเอกสารเสร็จสิ้น! เอกสารนี้ไม่เกี่ยวข้องกับ PDPA โปรดอัปโหลดเอกสารที่เกี่ยวข้องกับ PDPA เพื่อรับคำตอบเกี่ยวกับ PDPA")
            # แสดง PDF ในแถบด้านข้าง
            display_pdf(uploaded_file.getvalue(), uploaded_file.name)
    else:
        # ถ้าไม่มีไฟล์ถูกอัปโหลด ใช้ฐานความรู้
        if st.session_state.using_uploaded_file or st.session_state.langgraph_workflow is None:
            # คืนทรัพยากรจากไฟล์ที่อัปโหลด (ถ้ามี)
            if st.session_state.pdf_tool is not None:
                if hasattr(st.session_state.pdf_tool, "release_resources"):
                    st.session_state.pdf_tool.release_resources()
                st.session_state.pdf_tool = None
            
            # สร้าง workflow ใหม่สำหรับฐานความรู้
            st.session_state.langgraph_workflow = create_agents_and_tasks(
                st.session_state.knowledge_base_tool, 
                use_knowledge_base=True
            )
            st.session_state.using_uploaded_file = False
            st.session_state.is_pdpa_related = True  # ฐานความรู้ถือว่าเกี่ยวข้องกับ PDPA
        
        st.info("ไม่มีไฟล์ถูกอัปโหลด กำลังใช้ฐานความรู้ PDPA")

    st.button("ล้างการสนทนา", on_click=reset_chat)
    
    # แสดงสถานะของ web search
    st.markdown("---")
    st.markdown("### 🔧 การตั้งค่า")
    
    if SerperDevTool and os.getenv("SERPER_API_KEY"):
        st.success("🌐 Web Search: เปิดใช้งาน")
    elif SerperDevTool:
        st.warning("🌐 Web Search: ต้องการ API Key")
        st.info("เพิ่ม SERPER_API_KEY ใน .env file เพื่อเปิดใช้งาน web search")
    else:
        st.error("🌐 Web Search: ไม่พร้อมใช้งาน")
        st.info("ติดตั้ง serper_dev เพื่อเปิดใช้งาน web search")

# ===========================
#   แสดงประวัติการสนทนา
# ===========================
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else None):
        st.markdown(message["content"])

# ===========================
#   Chat Input
# ===========================
if st.session_state.using_uploaded_file:
    prompt_placeholder = "ถามคำถามเกี่ยวกับเอกสาร PDPA ที่อัปโหลด..."
else:
    prompt_placeholder = "ถามคำถามเกี่ยวกับ PDPA..."

prompt = st.chat_input(prompt_placeholder)

# Guardrail (UI layer) ก่อนเข้าสู่ workflow
if prompt:
    print(f"🔍 App: Processing prompt: {prompt}")
    
    # แสดงคำถามของผู้ใช้ก่อนที่จะตรวจสอบ SecurityFilter
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        if st.session_state.get("chat_store"):
            st.session_state.chat_store.add_message(
                session_id=st.session_state.session_id,
                role="user",
                content=prompt,
            )
    except Exception as e:
        st.warning(f"บันทึกประวัติผู้ใช้ไม่สำเร็จ: {e}")
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    _SecurityFilter = None
    try:
        from src.agentic_rag.tools.security_filter import SecurityFilter as _SecurityFilter
        print("✅ App: SecurityFilter imported successfully")
    except Exception as e:
        print(f"❌ App: SecurityFilter import failed: {e}")
        try:
            from agentic_rag.tools.security_filter import SecurityFilter as _SecurityFilter
            print("✅ App: SecurityFilter imported successfully (fallback)")
        except Exception as e2:
            print(f"❌ App: SecurityFilter import failed (fallback): {e2}")
            _SecurityFilter = None
    if _SecurityFilter is not None:
        try:
            print(f"🔍 SecurityFilter: Processing prompt: {prompt}")
            _ui_sf = _SecurityFilter()
            _ui_filter = _ui_sf.filter_user_input(prompt or "")
            print(f"🔍 SecurityFilter result: {_ui_filter}")
            
            if not _ui_filter.get("should_respond", True):
                print("🔴 SecurityFilter: BLOCKING prompt")
                st.session_state.messages.append({"role": "assistant", "content": _ui_filter.get("response_message") or "ตรวจพบเนื้อหาไม่เหมาะสมในคำถาม ⚠️ กรุณาพิมพ์ใหม่โดยใช้ถ้อยคำที่สุภาพ"})
                with st.chat_message("assistant"):
                    st.markdown(_ui_filter.get("response_message") or "ตรวจพบเนื้อหาไม่เหมาะสมในคำถาม ⚠️ กรุณาพิมพ์ใหม่โดยใช้ถ้อยคำที่สุภาพ")
                prompt = None
            else:
                print("✅ SecurityFilter: ALLOWING prompt")
        except Exception as e:
            # ถ้าตรวจไม่สำเร็จ ให้ข้ามและใช้ guardrail ระดับ workflow แทน
            print(f"❌ SecurityFilter error: {e}")
            st.error(f"SecurityFilter error: {e}")
            pass
    else:
        print("❌ SecurityFilter: Not available")

if prompt:
    # 2. รับการตอบกลับจาก LangGraph
    conversation_context = build_conversation_context(st.session_state.messages)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        
        # ตรวจสอบว่าไฟล์ไม่เกี่ยวข้องกับ PDPA และเราไม่ได้ใช้ฐานความรู้
        if st.session_state.using_uploaded_file and not st.session_state.is_pdpa_related:
            # ให้การตอบสนองเฉพาะสำหรับไฟล์ที่ไม่เกี่ยวข้องกับ PDPA
            full_response = "ขออภัย เอกสารที่อัปโหลดไม่เกี่ยวข้องกับ พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล (PDPA) ฉันสามารถให้คำตอบเฉพาะคำถามเกี่ยวกับ PDPA เท่านั้น โปรดอัปโหลดเอกสารที่เกี่ยวข้องกับ PDPA หรือถามคำถามเกี่ยวกับ PDPA ที่ฉันสามารถค้นคว้าจากฐานความรู้ของฉันได้"
            result = {"response": full_response}
        else:
            # รับการตอบสนองจาก LangGraph
            with st.spinner("กำลังคิด..."):
                print("\n" + "="*50)
                print(f"User Query: {prompt}")
                print("="*50 + "\n")
                print("🚀 LangGraph is kicking off the process...")
                conversation_history = f"Previous conversation:\n{conversation_context}\n\nNew question:"
                inputs = {"query": prompt, "context": conversation_history}
                # เปลี่ยนเป็น stream ทีละ step
                stream = st.session_state.langgraph_workflow.stream(inputs, stream_mode="values")
                progress_placeholder = st.empty()
                progress_log = []
                result = None
                last_with_answer = None
                for chunk in stream:
                    result = chunk
                    # อัปเดต progress ทีละบรรทัด
                    if "progress_log" in chunk and chunk["progress_log"]:
                        progress_log = chunk["progress_log"]
                        progress_placeholder.markdown(
                            "<div style='color: #888; opacity: 0.7; font-size: 0.92em;'>"
                            + "<br>".join([f"• {step}" for step in progress_log])
                            + "</div>", unsafe_allow_html=True
                        )
                    # เก็บ chunk ล่าสุดที่มี response หรือ candidates
                    if ("response" in chunk and chunk["response"]) or ("candidates" in chunk and chunk["candidates"]):
                        last_with_answer = chunk
                progress_placeholder.empty()
                if last_with_answer is not None:
                    result = last_with_answer
                print("\n" + "="*50)
                print("✅ LangGraph process finished.")
                print(f"🏁 Final Result: {result}")
                print("="*50 + "\n")
                # ฟังก์ชันช่วยดึงคำตอบที่ดีที่สุดจากผลลัพธ์
                def _extract_best_answer(res):
                    try:
                        if not isinstance(res, dict):
                            return ""
                        for key in ["response", "best_answer"]:
                            val = res.get(key)
                            if isinstance(val, str) and val.strip():
                                return val.strip()
                        for key in ["ranked", "candidates"]:
                            arr = res.get(key)
                            if isinstance(arr, list) and arr:
                                first_val = arr[0]
                                if isinstance(first_val, str) and first_val.strip():
                                    return first_val.strip()
                        return ""
                    except Exception:
                        return ""
                full_response = _extract_best_answer(result)
                if not full_response:
                    # fallback สุดท้ายป้องกันคำตอบว่าง
                    full_response = "ข้อมูลไม่เพียงพอในการสรุปคำตอบตาม PDPA โปรดระบุคำถามให้ชัดเจนหรืออัปโหลดเอกสารที่เกี่ยวข้องมากขึ้น"
        
        processing_time = time.time() - start_time
        
        # แสดงคำตอบที่ดีที่สุด (อันดับ 1) แทนที่จะเป็นคำตอบที่สังเคราะห์แล้ว
        best_answer = full_response
        if "candidates" in result and len(result["candidates"]) > 0:
            # ใช้คำตอบแรก (อันดับ 1) จาก candidates
            if isinstance(result["candidates"][0], str) and result["candidates"][0].strip():
                best_answer = result["candidates"][0].strip()
        
        # แสดง progress log (ถ้ามี)
        if "progress_log" in result and result["progress_log"]:
            with st.expander("🛠️ ขั้นตอนการคิด/ทำงานของ Agent (คลิกเพื่อดู)", expanded=False):
                for step in result["progress_log"]:
                    st.markdown(
                        f"""
                        <div style=\"margin-bottom: 8px; padding: 8px 12px; border-radius: 6px; color: #888; font-size: 0.97em;\">
                            {step}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        # แสดงการตอบสนองด้วยเอฟเฟกต์การพิมพ์
        lines = best_answer.split('\n') if isinstance(best_answer, str) else [str(best_answer)]
        for i, line in enumerate(lines):
            full_response_so_far = '\n'.join(lines[:i+1])
            message_placeholder.markdown(full_response_so_far + "▌")
            time.sleep(0.05)  # ปรับความเร็วตามต้องการ
        
        # แสดงการตอบสนองสุดท้ายโดยไม่มีเคอร์เซอร์
        message_placeholder.markdown(best_answer)
        
        # แสดงลิงก์อ้างอิงใน expander ถ้ามี
        if "web_references" in result and result["web_references"]:
            with st.expander("📚 แหล่งข้อมูลอ้างอิง (คลิกเพื่อดู/ซ่อน)", expanded=False):
                st.markdown(result["web_references"])
        

        
        # แสดงข้อมูลเพิ่มเติมเกี่ยวกับการประมวลผล
        info_container = st.container()
        with info_container:
            col1, col2, col3 = st.columns(3)
            with col1:
                if "retrieval_source" in result:
                    if result["retrieval_source"] == "pdf":
                        source_info = "📚 ฐานความรู้ PDPA"
                    elif result["retrieval_source"] == "pdf+web":
                        source_info = "🌐 ฐานความรู้ + เว็บ"
                        if "web_references" in result and result["web_references"]:
                            # นับจำนวนลิงก์อ้างอิง
                            ref_count = len([line for line in result["web_references"].split('\n') if line.strip()])
                            source_info += f" ({ref_count} แหล่ง)"
                    else:
                        source_info = "💭 คำตอบทั่วไป"
                    
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 6px; margin-top: 8px;">
                        <small style="color: #8f94fb;">{source_info}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 6px; margin-top: 8px;">
                        <small style="color: #8f94fb;">💭 คำตอบทั่วไป</small>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                if "candidates" in result and len(result["candidates"]) > 1:
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 6px; margin-top: 8px;">
                        <small style="color: #8f94fb;">🎯 สร้างคำตอบ {len(result["candidates"])} แบบ</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 6px; margin-top: 8px;">
                        <small style="color: #8f94fb;">💬 คำตอบเดียว</small>
                    </div>
                    """, unsafe_allow_html=True)
            with col3:
                # แสดงเวลาที่ใช้ในการประมวลผล
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 6px; margin-top: 8px;">
                    <small style="color: #8f94fb;">⏱️ ใช้เวลา {processing_time:.1f} วินาที</small>
                </div>
                """, unsafe_allow_html=True)
        
        # แสดงคำตอบอื่นๆ แบบเปิด-ปิดได้ (คล้าย ChatGPT)
        if "candidates" in result and len(result["candidates"]) > 1:
            st.markdown("---")
            with st.expander("🔍 ดูคำตอบอื่นๆ", expanded=False):
                st.markdown("""
                <div style="background: rgba(78, 84, 200, 0.1); padding: 12px; border-radius: 8px; margin-bottom: 16px;">
                    <p style="margin: 0; color: #8f94fb; font-size: 14px;">
                        💡 ระบบได้สร้างคำตอบหลายแบบให้คุณเลือก ด้านบนคือคำตอบที่ดีที่สุด 
                        คุณสามารถดูคำตอบอื่นๆ ได้ด้านล่างนี้
                    </p>
                </div>
                """, unsafe_allow_html=True)
                

                
                # แสดงคำตอบอื่นๆ
                for i, candidate in enumerate(result["candidates"][1:], 2):
                    st.markdown(f"""
                    <div class=\"alternative-answer\">
                        <h4>💡 คำตอบที่ {i}</h4>
                        <p>{candidate}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # 4. บันทึกข้อความของผู้ช่วยไปยังเซสชัน (ใช้คำตอบที่ดีที่สุด)
    st.session_state.messages.append({"role": "assistant", "content": best_answer})
    try:
        if st.session_state.get("chat_store"):
            st.session_state.chat_store.add_message(
                session_id=st.session_state.session_id,
                role="assistant",
                content=best_answer,
            )
    except Exception as e:
        st.warning(f"บันทึกประวัติผู้ช่วยไม่สำเร็จ: {e}")
    
    # 5. ทำ garbage collection ทุกรอบการสนทนา
    perform_periodic_gc()

# On Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
