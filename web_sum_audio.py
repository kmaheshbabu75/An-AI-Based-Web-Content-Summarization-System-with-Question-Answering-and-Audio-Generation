import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
import torch
import re
from gtts import gTTS
import os
from datetime import datetime
import base64

# Page configuration
st.set_page_config(
    page_title="Web Content Summarizer",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 20px 0;
    }
    .qa-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load and cache the models"""
    with st.spinner("🔄 Loading AI models... This may take a minute..."):
        # Load BART for summarization
        model_name = "sshleifer/distilbart-cnn-12-6"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Enable half precision for GPU
        if device == "cuda":
            model.half()
        
        # Load Question Answering model
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0 if device == "cuda" else -1
        )
        
        return tokenizer, model, device, qa_pipeline

def fetch_content(url):
    """Fetch and extract text content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if not text:
            raise ValueError("No content extracted from the page")
        
        return text
        
    except Exception as e:
        st.error(f"Error fetching content: {str(e)}")
        return None

def chunk_text(text, tokenizer, max_chunk_length=900):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(tokenizer.encode(word, add_special_tokens=False))
        
        if current_length + word_length > max_chunk_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_text(text, tokenizer, model, device, max_length=130, min_length=30):
    """Generate summary with optimized settings"""
    try:
        # Check if text fits in one chunk
        token_count = len(tokenizer.encode(text, truncation=False, add_special_tokens=False))
        
        if token_count <= 900:
            # Single chunk processing
            inputs = tokenizer.encode(
                text, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=2,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Multi-chunk processing
        chunks = chunk_text(text, tokenizer)
        summaries = []
        
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            inputs = tokenizer.encode(
                chunk, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=2,
                    early_stopping=True
                )
            
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
            progress_bar.progress((i + 1) / len(chunks))
        
        final_summary = ' '.join(summaries)
        
        # Re-summarize if needed
        if len(chunks) > 1 and len(final_summary.split()) > max_length:
            inputs = tokenizer.encode(
                final_summary, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=2,
                    early_stopping=True
                )
            
            final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        progress_bar.empty()
        return final_summary
        
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None

def text_to_audio(text):
    """Convert text to audio using gTTS"""
    try:
        # Create audio directory if it doesn't exist
        audio_dir = "audio_summaries"
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{timestamp}.mp3"
        filepath = os.path.join(audio_dir, filename)
        
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)
        
        return filepath
        
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def get_audio_player(filepath):
    """Create HTML audio player"""
    with open(filepath, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    return audio_html

def answer_question(question, context, qa_pipeline):
    """Answer a user's question based on the content"""
    try:
        result = qa_pipeline(question=question, context=context)
        return {
            'answer': result['answer'],
            'confidence': result['score']
        }
    except Exception as e:
        st.error(f"Error answering question: {str(e)}")
        return None

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">📝 Web Content Summarizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Summarize any webpage with AI-powered analysis, Q&A, and audio generation</div>', unsafe_allow_html=True)
    
    # Load models
    tokenizer, model, device, qa_pipeline = load_models()
    
    # Initialize session state
    if 'content' not in st.session_state:
        st.session_state.content = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Summary Settings")
        max_length = st.slider("Maximum Summary Length", 50, 300, 130, 10)
        min_length = st.slider("Minimum Summary Length", 20, 100, 30, 5)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info(f"**Device:** {device.upper()}\n\n**Summarization:** DistilBART\n\n**Q&A:** DistilBERT")
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Enter a webpage URL
        2. Click 'Summarize'
        3. Optionally convert to audio
        4. Ask questions about the content
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url = st.text_input("🔗 Enter Webpage URL:", placeholder="https://example.com/article")
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        summarize_button = st.button("🚀 Summarize", type="primary")
    
    # Process URL
    if summarize_button and url:
        if not url.startswith(('http://', 'https://')):
            st.error("⚠️ Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("🔍 Fetching content from webpage..."):
                content = fetch_content(url)
            
            if content:
                st.session_state.content = content
                
                # Show content preview
                with st.expander("📄 Original Content Preview", expanded=False):
                    st.write(f"**Total words:** {len(content.split())}")
                    st.write(f"**Characters:** {len(content)}")
                    st.text_area("Preview (first 1000 characters):", content[:1000] + "...", height=200)
                
                # Generate summary
                with st.spinner("🤖 Generating summary with AI..."):
                    summary = summarize_text(content, tokenizer, model, device, max_length, min_length)
                
                if summary:
                    st.session_state.summary = summary
                    
                    # Display summary
                    st.markdown("### 📋 Summary")
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Summary Words", len(summary.split()))
                    with col2:
                        st.metric("Compression Ratio", f"{len(content.split()) / len(summary.split()):.1f}x")
                    
                    st.success("✅ Summary generated successfully!")
    
    # Audio Generation Section
    if st.session_state.summary:
        st.markdown("---")
        st.markdown("### 🔊 Audio Generation")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            generate_audio = st.button("🎵 Generate Audio")
        
        if generate_audio:
            with st.spinner("🎙️ Converting summary to speech..."):
                audio_file = text_to_audio(st.session_state.summary)
            
            if audio_file:
                st.success("✅ Audio generated successfully!")
                audio_html = get_audio_player(audio_file)
                st.markdown(audio_html, unsafe_allow_html=True)
                
                # Download button
                with open(audio_file, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Audio",
                        data=f,
                        file_name=os.path.basename(audio_file),
                        mime="audio/mp3"
                    )
    
    # Q&A Section
    if st.session_state.content:
        st.markdown("---")
        st.markdown("### 💬 Ask Questions About the Content")
        
        question = st.text_input("🤔 Your Question:", placeholder="What is the main topic discussed?")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🔍 Ask")
        
        if ask_button and question:
            with st.spinner("🔎 Finding answer..."):
                result = answer_question(question, st.session_state.content, qa_pipeline)
            
            if result:
                st.session_state.qa_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'confidence': result['confidence']
                })
        
        # Display Q&A History
        if st.session_state.qa_history:
            st.markdown("#### 📝 Q&A History")
            for i, qa in enumerate(reversed(st.session_state.qa_history)):
                with st.container():
                    st.markdown(f'<div class="qa-box">', unsafe_allow_html=True)
                    st.markdown(f"**Q{len(st.session_state.qa_history) - i}:** {qa['question']}")
                    st.markdown(f"**A:** {qa['answer']}")
                    st.caption(f"Confidence: {qa['confidence']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🗑️ Clear Q&A History"):
                st.session_state.qa_history = []
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; padding: 20px;">'
        'Powered by DistilBART & DistilBERT | Made with Streamlit ❤️'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()