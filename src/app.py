"""
UIT RAG System - Streamlit Demo App.
Interactive RAG demo với hiển thị chi tiết quá trình xử lý.
"""

import logging
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.rag_utils import get_chunk_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Page config
st.set_page_config(
    page_title="UIT RAG Demo",
    page_icon="🎓",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .step-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #424242;
        padding: 0.5rem;
        background: linear-gradient(90deg, #E3F2FD, #BBDEFB);
        border-radius: 8px;
        margin-top: 1rem;
    }
    .step-pending { color: #9E9E9E; }
    .step-running { color: #FF9800; }
    .step-done { color: #4CAF50; }
    .step-error { color: #F44336; }
    .log-container {
        background: #FAFAFA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .log-entry {
        margin: 0.25rem 0;
        padding: 0.25rem 0;
        border-bottom: 1px solid #F5F5F5;
    }
    .answer-box {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-left: 4px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .context-box {
        background: #FFF8E1;
        border-left: 4px solid #FFC107;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background: #E3F2FD;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "chunks_loaded" not in st.session_state:
        st.session_state.chunks_loaded = False


def load_chunks_sidebar():
    """Load chunks in sidebar."""
    with st.sidebar:
        st.markdown("### 📚 Data Status")

        if not st.session_state.chunks_loaded:
            if st.button("🔄 Load Chunks", use_container_width=True):
                with st.spinner("Loading chunks..."):
                    db = get_chunk_database()
                    count = db.load()
                    st.session_state.chunks_loaded = True
                    st.success(f"✅ Loaded {count} chunks!")
                    st.rerun()
        else:
            db = get_chunk_database()
            st.success(f"✅ {len(db.chunks)} chunks loaded")

        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        st.info(f"Model: DeepSeek V4 Flash")
        st.info("Provider: DeepSeek via xah.io")


def render_step_indicator(step, step_index):
    """Render a step with status indicator."""
    status_colors = {
        "pending": "⚪",
        "running": "🟡",
        "done": "🟢",
        "error": "🔴",
        "skipped": "⚪",
    }
    status_text = {
        "pending": "Chờ xử lý",
        "running": "Đang xử lý...",
        "done": "Hoàn thành",
        "error": "Lỗi",
        "skipped": "Bỏ qua",
    }

    icon = status_colors.get(step.status, "⚪")
    status_name = status_text.get(step.status, step.status)

    with st.expander(f"{icon} **Step {step_index + 1}: {step.name}** - {status_name}", expanded=(step.status in ["running", "done", "error"])):
        if step.log_messages:
            st.markdown("**Logs:**")
            for msg in step.log_messages:
                if "ERROR" in msg or "Lỗi" in msg:
                    st.error(msg)
                elif "completed" in msg.lower() or "hoàn thành" in msg.lower():
                    st.success(msg)
                else:
                    st.text(msg)

        if step.input_data:
            st.markdown("**Input:**")
            st.json(step.input_data)

        if step.output_data:
            st.markdown("**Output:**")
            st.json(step.output_data)

        if step.error:
            st.error(f"Error: {step.error}")

        if step.start_time and step.end_time:
            duration = (step.end_time - step.start_time) * 1000
            st.caption(f"⏱️ Duration: {duration:.0f}ms")


def render_answer(result):
    """Render the final answer."""
    st.markdown("---")
    st.markdown("## 🎯 Kết quả")

    # Answer card
    st.markdown('<div class="answer-box">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("UIT Related", "✅ Có" if result.is_about_uit else "❌ Không")

    with col2:
        total_time = sum(s.duration_ms() for s in result.steps)
        st.metric("Total Time", f"{total_time:.0f}ms")

    st.markdown("### Câu trả lời:")
    st.markdown(f"**{result.answer}**")

    if result.explanation:
        st.markdown("### Giải thích:")
        st.info(result.explanation)

    st.markdown('</div>', unsafe_allow_html=True)


def render_context_viewer(result):
    """Render context viewer."""
    st.markdown("### 📄 Chi tiết Context")

    for i, step in enumerate(result.steps):
        if step.name in ["Hybrid Search", "Re-Retrieval"] and step.output_data:
            chunks = step.output_data.get("chunks", [])
            if chunks:
                st.markdown(f"#### Ngữ cảnh từ bước: **{step.name}**")
                for r in chunks:
                    with st.expander(f"📄 Chunk: {r.get('chunk_id', 'Unknown')}"):
                        st.markdown(f"**Category:** {r.get('category', 'N/A')}")
                        st.markdown(f"**Year:** {r.get('year', 'N/A')}")
                        st.markdown(f"**Major:** {r.get('major', 'N/A')}")
                        st.markdown(f"**Content:**")
                        st.text(r.get("content", ""))


def main():
    init_session_state()

    # Header
    st.markdown('<h1 class="main-header">🎓 UIT RAG System Demo</h1>', unsafe_allow_html=True)
    st.markdown("### Tra cứu thông tin về Nội quy & Quy chế UIT")

    load_chunks_sidebar()

    # Main content
    st.markdown("### 💬 Đặt câu hỏi")

    mode = st.radio(
        "Chế độ:",
        ["normal", "mcq"],
        format_func=lambda x: "💬 Normal" if x == "normal" else "📝 MCQ (Trắc nghiệm)",
        horizontal=True,
    )

    question = st.text_area(
        "Câu hỏi:",
        placeholder="VD: Học phí ngành Khoa học Máy tính năm 2026 là bao nhiêu?",
        height=100,
    )

    # MCQ options
    options = None
    if mode == "mcq":
        st.markdown("**Đáp án (MCQ - 5 đáp án):**")
        sub_col1, sub_col2, sub_col3 = st.columns(3)
        with sub_col1:
            opt_a = st.text_input("A:", key="opt_a")
            opt_b = st.text_input("B:", key="opt_b")
        with sub_col2:
            opt_c = st.text_input("C:", key="opt_c")
            opt_d = st.text_input("D:", key="opt_d")
        with sub_col3:
            opt_e = st.text_input("E:", key="opt_e")

        options = {}
        if opt_a:
            options["A"] = opt_a
        if opt_b:
            options["B"] = opt_b
        if opt_c:
            options["C"] = opt_c
        if opt_d:
            options["D"] = opt_d
        if opt_e:
            options["E"] = opt_e

    submitted = st.button("🚀 Tra cứu", type="primary", use_container_width=True)

    # Run pipeline
    if submitted and question:
        with st.spinner("🔍 Đang xử lý..."):
            pipeline = RAGPipeline()
            result = pipeline.run(question, mode, options)

            st.session_state.current_result = result

            if result.success:
                if not result.is_about_uit:
                    st.warning("⚠️ Câu hỏi không liên quan đến UIT")
                else:
                    st.success("✅ Xử lý hoàn tất!")
                
                # Append to history
                st.session_state.history.append({
                    "question": question,
                    "mode": mode,
                    "answer": result.answer,
                })
            else:
                st.error(f"❌ Có lỗi xảy ra: {result.error}")

            st.rerun()

    # Render answer
    if st.session_state.current_result:
        result = st.session_state.current_result
        render_answer(result)

        # Tabs for detailed view
        tab_progress, tab_context, tab_history = st.tabs([
            "⚙️ Quá trình & Logs", 
            "📄 Context", 
            "📜 Lịch sử"
        ])

        with tab_progress:
            st.markdown("### 📊 Pipeline Status")
            # Progress bar
            completed_steps = sum(1 for s in result.steps if s.status == "done")
            total_steps = len(result.steps)
            progress = completed_steps / total_steps
            st.progress(progress, text=f"Progress: {completed_steps}/{total_steps} steps completed")

            # Render all steps
            for i, step in enumerate(result.steps):
                render_step_indicator(step, i)

            st.markdown("---")
            st.markdown("### 📋 Chi tiết Logs từng bước")
            for step_name, logs in result.get_all_logs():
                if logs.startswith("==="):
                    st.markdown(f"**{logs}**")
                else:
                    if "ERROR" in logs or "Lỗi" in logs:
                        st.error(f"`{logs}`")
                    elif "✅" in logs or "completed" in logs.lower():
                        st.success(f"`{logs}`")
                    else:
                        st.text(f"  {logs}")

        with tab_context:
            render_context_viewer(result)

        with tab_history:
            st.markdown("### 📜 Lịch sử tra cứu")

            if st.session_state.history:
                for i, item in enumerate(st.session_state.history):
                    with st.expander(f"Q{i+1}: {item['question'][:50]}..."):
                        st.markdown(f"**Mode:** {item['mode']}")
                        st.markdown(f"**Answer:** {item['answer']}")
            else:
                st.info("Chưa có lịch sử tra cứu")


if __name__ == "__main__":
    main()
