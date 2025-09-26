import os
import yaml
from .tools.custom_tool import DocumentSearchTool
from .tools.qdrant_storage import QdrantStorage, MyEmbedder
from langgraph.graph import StateGraph
import logging
from typing import Dict, Any
# Security filter for guardrails
from .tools.security_filter import SecurityFilter
# Import SerperDevTool for web search
try:
    from .tools.serper_tool import SerperDevTool
except ImportError:
    SerperDevTool = None
    logging.warning("SerperDevTool not installed. Please check serper_tool.py if you want web search.")

# Add OpenAI-compatible client (for llama.cpp server)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logging.warning("OpenAI client not installed. Please install with 'pip install openai'.")

OLLAMA_MODEL = "hf.co/scb10x/typhoon2.1-gemma3-4b-gguf:Q4_K_M"
LLAMA_CPP_BASE_URL = "http://localhost:8080/v1"

AGENTS_YAML = os.path.join(os.path.dirname(__file__), 'config', 'agents.yaml')
TASKS_YAML = os.path.join(os.path.dirname(__file__), 'config', 'tasks.yaml')

# Helper to call llama.cpp (OpenAI-compatible) LLM

def call_llm(prompt, system=None):
    if OpenAI is None:
        raise ImportError("OpenAI client not installed. Run: pip install openai")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    client = OpenAI(base_url=LLAMA_CPP_BASE_URL, api_key="not-needed")
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        max_tokens=8192
    )
    return response.choices[0].message.content


def build_langgraph_workflow(pdf_tool=None, use_knowledge_base=True):
    with open(AGENTS_YAML, 'r', encoding='utf-8') as f:
        agents_config = yaml.safe_load(f)
    with open(TASKS_YAML, 'r', encoding='utf-8') as f:
        tasks_config = yaml.safe_load(f)

    # Initialize web search tool if available and API key is set
    web_search_tool = None
    print(f"🔍 Checking SerperDevTool availability...")
    print(f"   - SerperDevTool imported: {SerperDevTool is not None}")
    print(f"   - SERPER_API_KEY exists: {os.getenv('SERPER_API_KEY') is not None}")
    
    if SerperDevTool and os.getenv("SERPER_API_KEY"):
        try:
            web_search_tool = SerperDevTool()
            print("✅ SerperDevTool initialized successfully for LangGraph")
            # Debug: ดู method ที่มี
            print(f"🔍 SerperDevTool methods: {[m for m in dir(web_search_tool) if not m.startswith('_')]}")
        except Exception as e:
            print(f"❌ Error initializing SerperDevTool: {e}")
            web_search_tool = None
    else:
        if not SerperDevTool:
            print("⚠️ SerperDevTool not available - install serper-dev")
        if not os.getenv("SERPER_API_KEY"):
            print("⚠️ SERPER_API_KEY not set - add to .env file")
        print("⚠️ Web search will be disabled")

    # Initialize security filter (guardrail)
    security_filter = SecurityFilter()

    # --- Node implementations ---
    def append_progress(state, message):
        progress = state.get("progress_log", [])
        return progress + [message]

    def refine_question_node(state):
        query = state.get("query", "")
        progress_log = state.get("progress_log", [])
        # Guardrail: ตรวจสอบคำถามก่อน refine หากพบคำหยาบ/ไม่เหมาะสม ให้หยุดและแจ้งเตือน
        try:
            filter_result = security_filter.filter_user_input(query or "")
        except Exception:
            filter_result = {"should_respond": False, "response_message": "เกิดข้อผิดพลาดในการตรวจสอบความปลอดภัยของข้อความ กรุณาลองใหม่อีกครั้ง"}

        if not filter_result.get("should_respond", True):
            progress_log = append_progress({"progress_log": progress_log}, "🔴 [Guardrail] บล็อกคำถามเนื่องจากพบคำหยาบ/ไม่เหมาะสม")
            warn_msg = filter_result.get("response_message") or "ตรวจพบเนื้อหาไม่เหมาะสมในคำถาม ⚠️ กรุณาพิมพ์ใหม่โดยใช้ถ้อยคำที่สุภาพ"
            return {**state, "response": warn_msg, "best_answer": "", "blocked": True, "progress_log": progress_log}

        # ผ่านการตรวจสอบความปลอดภัยแล้ว จึงเริ่ม refineห
        progress_log = append_progress({"progress_log": progress_log}, "🟡 [LangGraph] กำลังปรับคำถาม (Refining question)...")
        system = agents_config['question_refiner_agent']['role'] + "\n" + agents_config['question_refiner_agent']['goal']
        prompt = (
            f"Refine or clarify the following question to make it clear, specific, and actionable.\n"
            f"Question: {query}\n"
            f"โปรดตอบเป็นภาษาไทยเท่านั้น"
        )
        refined = call_llm(prompt, system=system)
        progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] ปรับคำถามเสร็จแล้ว (Refined question)")
        return {**state, "refined_question": refined, "progress_log": progress_log}

    def planning_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] กำลังวางแผน (Planning)...")
        refined = state.get("refined_question", "")
        system = agents_config['planning_agent']['role'] + "\n" + agents_config['planning_agent']['goal']
        prompt = (
            f"Generate a step-by-step plan to answer the following question.\n"
            f"Question: {refined}\n"
            f"โปรดตอบเป็นภาษาไทยเท่านั้น"
        )
        plan = call_llm(prompt, system=system)
        progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] วางแผนเสร็จแล้ว (Planning done)")
        return {**state, "plan": plan, "progress_log": progress_log}

    def retrieval_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] กำลังค้นข้อมูล (Retrieving from PDF/Knowledge)...")
        query = state.get("query", "")  # ใช้ query เดิม ไม่ใช้ refined_question
        tool = pdf_tool if pdf_tool else DocumentSearchTool(file_path=os.path.join(os.path.dirname(__file__), '../../knowledge/pdpa.pdf'))
        retrieved = tool._run(query)
        try:
            print("\n===== DocumentSearchTool Result (truncated) =====")
            if isinstance(retrieved, str):
                preview = retrieved[:2000]
                print(preview)
                if len(retrieved) > len(preview):
                    print(f"... [truncated, total {len(retrieved)} chars]")
            else:
                print(str(retrieved))
            print("===== End DocumentSearchTool Result =====\n")
        except Exception as _:
            pass
        progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] ค้นข้อมูลเสร็จแล้ว (Retrieval done)")
        return {**state, "retrieved": retrieved, "retrieval_source": "pdf", "progress_log": progress_log}

    def websearch_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] กำลังค้นเว็บ (Web search fallback)...")
        query = state.get("query", "")  # ใช้ query เดิม ไม่ใช้ refined_question
        web_text = ""
        references_text = ""  # เพิ่มบรรทัดนี้เพื่อป้องกัน error
        
        # Helper: keep only readable characters and cap length
        import re
        def _clean_text(text: str, max_len: int = 4000) -> str:
            if not isinstance(text, str):
                text = str(text)
            # Remove binary-looking sequences and non-printable chars
            text = re.sub(r"[^\t\n\r\x20-\x7E\u0E00-\u0E7F\u2013\u2014\u2018\u2019\u201C\u201D]", " ", text)
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > max_len:
                return text[:max_len] + " …(ตัดทอน)"
            return text
        
        if web_search_tool:
            try:
                print(f"🔍 Trying to call SerperDevTool with query: '{query}'")
                print(f"🔍 Available methods: {[m for m in dir(web_search_tool) if not m.startswith('_')]}")
                
                # ตรวจสอบ schema ของ tool
                if hasattr(web_search_tool, 'args_schema'):
                    print(f"🔍 Tool args_schema: {web_search_tool.args_schema}")
                    if hasattr(web_search_tool.args_schema, '__annotations__'):
                        print(f"🔍 Tool annotations: {web_search_tool.args_schema.__annotations__}")
                if hasattr(web_search_tool, 'schema'):
                    print(f"🔍 Tool schema: {web_search_tool.schema}")
                
                # ตรวจสอบ signature ของ run method
                import inspect
                if hasattr(web_search_tool, 'run'):
                    try:
                        sig = inspect.signature(web_search_tool.run)
                        print(f"🔍 run method signature: {sig}")
                    except Exception as e:
                        print(f"🔍 Cannot inspect run method: {e}")
                
                # ลองใช้ method ที่ถูกต้อง
                web_result = None
                
                # ลองใช้ run method แบบ keyword argument ก่อน
                if hasattr(web_search_tool, 'run'):
                    print("🔍 Trying run method with keyword argument")
                    try:
                        web_result = web_search_tool.run(query=query)
                        print("✅ run method with keyword argument succeeded")
                    except Exception as e:
                        print(f"🔍 run method with keyword failed: {e}")
                        try:
                            # ลองใช้ run method แบบ positional argument
                            web_result = web_search_tool.run(query)
                            print("✅ run method with positional argument succeeded")
                        except Exception as e2:
                            print(f"🔍 run method with positional failed: {e2}")
                
                # ถ้ายังไม่ได้ ลองใช้ _run method
                if web_result is None and hasattr(web_search_tool, '_run'):
                    print("🔍 Trying _run method")
                    try:
                        web_result = web_search_tool._run(query)
                        print("✅ _run method succeeded")
                    except Exception as e:
                        print(f"🔍 _run method failed: {e}")
                
                # ถ้ายังไม่ได้ ลองเรียกใช้โดยตรง
                if web_result is None:
                    print("🔍 Trying direct call")
                    try:
                        web_result = web_search_tool.search(query)
                        print("✅ .search method succeeded")
                    except Exception as e:
                        print(f"🔍 .search method failed: {e}")
                
                # ถ้าไม่สามารถเรียกใช้ได้เลย
                if web_result is None:
                    raise Exception("ไม่สามารถเรียกใช้ SerperDevTool ได้")
                
                if isinstance(web_result, dict) and 'organic' in web_result:
                    # สร้างข้อความและลิงก์อ้างอิง
                    results = web_result['organic']
                    web_text_parts = []
                    references = []
                    web_contents = []
                    combined_chars = 0
                    combined_char_budget = 12000
                    for i, result in enumerate(results):  # ดึงเนื้อหาทุกลิงก์
                        title = result.get('title', '')
                        snippet = result.get('snippet', '')
                        link = result.get('link', '')
                        print(f"🌐 [{i+1}] อ่านลิงก์: {link}")
                        web_text_parts.append(f"{i+1}. {title}\n{_clean_text(snippet, 600)}")
                        references.append(f"[{i+1}] {title}: {link}")
                        # ดึงเนื้อหาทั้งหมดจากเว็บ
                        from src.agentic_rag.tools.serper_tool import SerperDevTool
                        content = SerperDevTool.extract_web_content(link, max_chars=200000)
                        cleaned = _clean_text(content, 4000)
                        combined_chars += len(cleaned)
                        print(f"    ↳ ความยาวหลังทำความสะอาด: {len(cleaned)} ตัวอักษร")
                        # Append only within budget to avoid overwhelming LLM
                        if combined_chars <= combined_char_budget:
                            web_contents.append(f"---\n{title}\n{link}\n{cleaned}\n")
                        else:
                            web_contents.append(f"---\n{title}\n{link}\n(ตัดทอนเนื้อหาเพื่อจำกัดขนาด)\n")
                    web_text = '\n\n'.join(web_text_parts) + '\n\n' + '\n'.join(web_contents)
                    references_text = '\n'.join(references)
                    print(f"✅ [LangGraph] Web search successful, found {len(results)} results")
                    print(f"📚 References: {len(references)} sources")
                    # Do not print web_text to avoid huge console noise / binary
                else:
                    web_text = str(web_result)
                    references_text = ""
                    print(f"✅ [LangGraph] Web search successful, raw result")
            except Exception as e:
                print(f"❌ [LangGraph] Web search error: {e}")
                web_text = "ไม่สามารถค้นเว็บได้"
        else:
            web_text = "ไม่สามารถค้นเว็บได้ (SerperDevTool ไม่พร้อมใช้งานหรือไม่มี API Key)"
            print("⚠️ [LangGraph] Web search not available")
        
        # Combine with previous retrieved
        combined = f"[PDF/Knowledge]: {state.get('retrieved', '')}\n[Web]: {web_text}"
        progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] ค้นเว็บเสร็จแล้ว (Web search done)")
        return {**state, "retrieved": combined, "retrieval_source": "pdf+web", "web_search_count": state.get("web_search_count", 0) + 1, "web_references": references_text, "progress_log": progress_log}

    def judge_info_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] LLM ประเมินความเพียงพอของข้อมูล (Judging info sufficiency)...")
        refined = state.get("refined_question", "")
        context = state.get("retrieved", "")
        
        # ตรวจสอบจำนวนครั้งที่พยายามค้นหาแล้ว
        web_search_count = state.get("web_search_count", 0)
        if web_search_count >= 3:
            print("🟡 [LangGraph] เกินจำนวนครั้งที่พยายามค้นหาแล้ว - จะใช้ข้อมูลที่มี")
            return {**state, "info_sufficient": True, "judge_reason": "ใช้ข้อมูลที่มีหลังจากพยายามค้นหาแล้ว", "progress_log": progress_log}
        
        # ตรวจสอบว่าข้อมูลมีเนื้อหาที่เป็นประโยชน์หรือไม่
        if not context or context.strip() in ["ไม่พบผลลัพธ์ที่เกี่ยวข้อง", "โปรดตั้งคำถามเฉพาะเกี่ยวกับ PDPA เท่านั้น", "ไม่สามารถค้นเว็บได้"]:
            print("🟡 [LangGraph] ข้อมูลไม่เพียงพอ - จะใช้ web search")
            return {**state, "info_sufficient": False, "judge_reason": "ข้อมูลไม่เพียงพอ", "web_search_count": web_search_count + 1, "progress_log": progress_log}
        
        system = "คุณเป็นผู้ช่วยที่เชี่ยวชาญในการประเมินความครบถ้วนของข้อมูลสำหรับการตอบคำถาม"
        prompt = (
            f"คำถาม: {refined}\n"
            f"ข้อมูลที่ค้นพบ: {context}\n"
            f"ข้อมูลนี้เพียงพอสำหรับการตอบคำถามหรือไม่?\n"
            f"ถ้าเพียงพอ ตอบว่า 'เพียงพอ'\nถ้าไม่เพียงพอ ตอบว่า 'ไม่เพียงพอ' และระบุว่าข้อมูลขาดอะไร\nโปรดตอบเป็นภาษาไทยเท่านั้น"
        )
        judge = call_llm(prompt, system=system)
        progress_log = append_progress({"progress_log": progress_log}, f"🟢 [LangGraph] LLM ประเมินแล้ว: {judge.strip()}")
        # Simple logic: if 'เพียงพอ' in answer and not 'ไม่เพียงพอ' => sufficient
        is_sufficient = ('เพียงพอ' in judge and 'ไม่เพียงพอ' not in judge)
        return {**state, "info_sufficient": is_sufficient, "judge_reason": judge.strip(), "progress_log": progress_log}

    def generate_answers_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] กำลังสร้างคำตอบหลายแบบ (Generating multiple answers)...")
        refined = state.get("refined_question", "")
        context = state.get("retrieved", "")
        system = agents_config['answer_candidate_agent']['role'] + "\n" + agents_config['answer_candidate_agent']['goal']

        num_candidates = 3
        candidates = []
        for i in range(num_candidates):
            prompt = (
                f"Using the following context, write ONE comprehensive, structured answer to the question.\n"
                f"Context: {context}\n"
                f"Question: {refined}\n"
                f"\nข้อกำหนด:\n"
                f"- ต้องเป็นการวิเคราะห์เชิงกฎหมายภายใต้ PDPA ของไทยเท่านั้น\n"
                f"- หากข้อมูลไม่เพียงพอ ระบุว่า 'ข้อมูลไม่เพียงพอ' และแนะนำทางปฏิบัติ\n"
                f"- ตอบครบทุกประเด็นของคำถามและอ้างอิงมาตราอย่างชัดเจน\n"
                f"- จัดรูปแบบเป็นหัวข้อย่อย กระชับ อ่านง่าย (ภาษาไทย)\n"
                f"- คำตอบนี้ต้องมีมุมมองหรือโครงสร้างที่แตกต่างจากคำตอบอื่น ๆ (หากมี)\n"
                f"\nอย่าอ้างอิงถึงคำตอบอื่น และสร้างคำตอบเพียง 1 ชุดเท่านั้น\n"
            )
            try:
                answer = call_llm(prompt, system=system).strip()
            except Exception as e:
                answer = f"ไม่สามารถสร้างคำตอบลำดับที่ {i+1} ได้: {e}"
            candidates.append(answer)

        progress_log = append_progress({"progress_log": progress_log}, f"🟢 [LangGraph] สร้างคำตอบเสร็จแล้ว {len(candidates)} แบบ (Candidates ready)")
        return {**state, "candidates": candidates, "progress_log": progress_log}

    def decision_ranking_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] จัดอันดับคำตอบ (Ranking candidates)...")
        candidates = state.get("candidates", [])
        refined = state.get("refined_question", "")
        if not candidates:
            progress_log = append_progress({"progress_log": progress_log}, "🟡 [LangGraph] ไม่มี candidates สำหรับจัดอันดับ")
            return {**state, "ranked": [], "best_answer": state.get("best_answer", ""), "progress_log": progress_log}

        system = agents_config['decision_ranking_agent']['role'] + "\n" + agents_config['decision_ranking_agent']['goal']
        indexed = "\n".join([f"[{i+1}]\n{c}" for i, c in enumerate(candidates)])
        prompt = (
            f"Evaluate the following candidate answers for the question and return ONLY a comma-separated list of indices from best to worst (e.g., 2,1,3).\n"
            f"Question: {refined}\n"
            f"Candidates:\n{indexed}\n"
            f"\nตอบเฉพาะหมายเลขดัชนีคั่นด้วยจุลภาคเท่านั้น (เช่น 2,1,3) เป็นภาษาไทยหรืออังกฤษก็ได้ แต่ห้ามมีข้อความอื่น"
        )
        order_text = call_llm(prompt, system=system)
        import re
        nums = re.findall(r"\d+", order_text)
        order = [int(n)-1 for n in nums if 1 <= int(n) <= len(candidates)]
        # Ensure we have a full permutation; append any missing indices in original order
        missing = [i for i in range(len(candidates)) if i not in order]
        order.extend(missing)

        ranked = [candidates[i] for i in order]
        best_answer = ranked[0] if ranked else ""
        progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] จัดอันดับคำตอบเสร็จแล้ว (Ranking done)")
        return {**state, "ranked": ranked, "candidates": ranked, "best_answer": best_answer, "progress_log": progress_log}

    def response_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] กำลังสรุปคำตอบ (Synthesizing response)...")
        ranked = state.get("ranked", [])
        best_answer = state.get("best_answer", "")
        web_references = state.get("web_references", "")
        
        if best_answer:
            # จัดรูปแบบคำตอบให้กระชับ อ่านง่าย และครบประเด็น
            system = agents_config['response_synthesizer_agent']['role'] + "\n" + agents_config['response_synthesizer_agent']['goal']
            prompt = (
                f"จัดรูปแบบคำตอบต่อไปนี้ให้กระชับ เป็นหัวข้อย่อยอ่านง่าย ครอบคลุมทุกคำถามย่อย และไม่เพิ่มเนื้อหาใหม่:\n"
                f"คำตอบเดิม: {best_answer}\n"
                f"\n⚠️ กฎสำคัญ:\n"
                f"- ต้องตอบให้ถูกต้องตาม พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562 เท่านั้น\n"
                f"- หากข้อมูลไม่เพียงพอ ให้ระบุว่า 'ข้อมูลไม่เพียงพอ' และแนะนำให้ปรึกษาผู้เชี่ยวชาญ\n"
                f"- ใช้ bullet points (•) และหัวข้อย่อยให้ชัดเจน\n"
                f"- เน้นความกระชับ อ่านง่าย และครบประเด็น\n"
                f"- ไม่เพิ่มข้อมูลใหม่ที่ไม่มีในคำตอบเดิม\n"
                f"\nโปรดตอบเป็นภาษาไทยเท่านั้น"
            )
            response = call_llm(prompt, system=system)
        else:
            # Fallback to synthesizing from ranked answers
            system = agents_config['response_synthesizer_agent']['role'] + "\n" + agents_config['response_synthesizer_agent']['goal']
            prompt = (
                f"Select the top-ranked answer and format it as the final response.\n"
                f"Answers: {ranked}\n"
                f"\n⚠️ กฎสำคัญ: ต้องตอบให้ถูกต้องตาม พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562 เท่านั้น\n"
                f"โปรดตอบเป็นภาษาไทยเท่านั้น"
            )
            response = call_llm(prompt, system=system)
        
        progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] สรุปคำตอบเสร็จแล้ว (Response ready)")
        return {**state, "response": response, "best_answer": best_answer, "web_references": web_references, "progress_log": progress_log}

    # --- Build the graph ---
    graph = StateGraph(Dict[str, Any])
    graph.add_node("refine_question", refine_question_node)
    graph.add_node("planning", planning_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("websearch", websearch_node)
    graph.add_node("judge_info", judge_info_node)
    graph.add_node("generate_answers", generate_answers_node)
    graph.add_node("decision_ranking", decision_ranking_node)
    graph.add_node("response", response_node)

    # Wiring: retrieval -> judge_info
    graph.set_entry_point("refine_question")
    graph.add_conditional_edges(
        "refine_question",
        lambda state: "response" if state.get("blocked") else "planning"
    )
    graph.add_edge("planning", "retrieval")
    graph.add_edge("retrieval", "judge_info")
    # If info sufficient, go to generate_answers; else, go to websearch
    graph.add_conditional_edges(
        "judge_info",
        lambda state: "generate_answers" if state.get("info_sufficient") else "websearch"
    )
    # After websearch, judge again
    graph.add_edge("websearch", "judge_info")
    # After info is sufficient, continue with ranking then response
    graph.add_edge("generate_answers", "decision_ranking")
    graph.add_edge("decision_ranking", "response")
    graph.set_finish_point("response")

    return graph.compile()
