import pickle
import faiss
from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import hashlib
import re
import os
import sys

# Import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("❌ ERROR: groq package not installed!")
    print("Add 'groq' to requirements.txt")
    GROQ_AVAILABLE = False
    sys.exit(1)

app = Flask(__name__)

print("=" * 50)
print("STARTING MCQ GENERATOR APP")
print("=" * 50)

# ------------------------------
# Initialize Groq API Client
# ------------------------------
print("\nStep 1: Checking Groq API Key...")
print("-" * 50)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found!")
    print("\nTo fix this:")
    print("1. Go to: https://console.groq.com/keys")
    print("2. Create a free API key")
    print("3. In HuggingFace Space Settings → Repository secrets")
    print("   Add: Name=GROQ_API_KEY, Value=<your-key>")
    print("4. Restart your Space")
    groq_client = None
else:
    print(f"✓ GROQ_API_KEY found ({len(GROQ_API_KEY)} chars)")
    print(f"  First 20 chars: {GROQ_API_KEY[:20]}...")
    
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Test the API
        print("  Testing API connection...")
        test = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="llama-3.3-70b-versatile",
            max_tokens=5
        )
        print("✓ Groq API working!")
        
    except Exception as e:
        print(f"❌ Groq API initialization failed:")
        print(f"   Error: {str(e)}")
        groq_client = None

print("-" * 50)

# ------------------------------
# Load embedding model (CPU)
# ------------------------------
print("\nStep 2: Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ Embedding model loaded")

# ------------------------------
# Download files from Hugging Face
# ------------------------------
REPO_ID = "Redfire-1234/pcb_tutor"

print("\nStep 3: Downloading subject files...")
print("-" * 50)

try:
    bio_chunks_path = hf_hub_download(repo_id=REPO_ID, filename="bio_chunks.pkl", repo_type="model")
    faiss_bio_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_bio.bin", repo_type="model")
    
    chem_chunks_path = hf_hub_download(repo_id=REPO_ID, filename="chem_chunks.pkl", repo_type="model")
    faiss_chem_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_chem.bin", repo_type="model")
    
    phy_chunks_path = hf_hub_download(repo_id=REPO_ID, filename="phy_chunks.pkl", repo_type="model")
    faiss_phy_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_phy.bin", repo_type="model")
    
    print("✓ All files downloaded")
except Exception as e:
    print(f"❌ Error downloading files: {e}")
    sys.exit(1)

# Load all subjects into memory
print("\nStep 4: Loading subject data into memory...")
SUBJECTS = {
    "biology": {
        "chunks": pickle.load(open(bio_chunks_path, "rb")),
        "index": faiss.read_index(faiss_bio_path)
    },
    "chemistry": {
        "chunks": pickle.load(open(chem_chunks_path, "rb")),
        "index": faiss.read_index(faiss_chem_path)
    },
    "physics": {
        "chunks": pickle.load(open(phy_chunks_path, "rb")),
        "index": faiss.read_index(faiss_phy_path)
    }
}

print(f"✓ Biology: {len(SUBJECTS['biology']['chunks'])} chunks")
print(f"✓ Chemistry: {len(SUBJECTS['chemistry']['chunks'])} chunks")
print(f"✓ Physics: {len(SUBJECTS['physics']['chunks'])} chunks")

print("\n" + "=" * 50)
print("✓ ALL SYSTEMS READY!")
print("=" * 50 + "\n")

# ------------------------------
# Caching
# ------------------------------
MCQ_CACHE = {}
MAX_CACHE_SIZE = 100

def get_cache_key(topic, subject, context_hash):
    return f"{subject}:{topic}:{context_hash}"

def cache_mcq(key, mcqs):
    if len(MCQ_CACHE) >= MAX_CACHE_SIZE:
        MCQ_CACHE.pop(next(iter(MCQ_CACHE)))
    MCQ_CACHE[key] = mcqs

# ------------------------------
# RAG Search
# ------------------------------
def rag_search(query, subject, k=5):
    if subject not in SUBJECTS:
        return None
    
    chunks = SUBJECTS[subject]["chunks"]
    index = SUBJECTS[subject]["index"]
    
    q_emb = embed_model.encode([query], show_progress_bar=False).astype("float32")
    D, I = index.search(q_emb, k)
    
    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    
    return "\n\n".join(results)

# ------------------------------
# MCQ Generation
# ------------------------------
def generate_mcqs(context, topic, subject):
    # Check if Groq is available
    if not groq_client:
        error_msg = """ERROR: Groq API not initialized!
Please check:
1. GROQ_API_KEY is set in Space Settings → Repository secrets
2. API key is valid (get one from https://console.groq.com/keys)
3. Space has been restarted after adding the key
Current status: API key not found or invalid."""
        return error_msg
    
    # Check cache
    context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
    cache_key = get_cache_key(topic, subject, context_hash)
    
    if cache_key in MCQ_CACHE:
        print("✓ Using cached MCQs")
        return MCQ_CACHE[cache_key]
    
    print(f"🤖 Generating MCQs for {subject} - {topic}")
    
    prompt = f"""You are a Class-12 {subject.title()} teacher creating MCQs.
Topic: "{topic}"
Reference material from textbook:
{context[:1500]}
Generate exactly 5 multiple-choice questions based on the reference material.
FORMAT (follow EXACTLY):
Q1. [Question based on material]
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
Answer: [A/B/C/D] - [Brief explanation]
Q2. [Question based on material]
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
Answer: [A/B/C/D] - [Brief explanation]
Continue for Q3, Q4, Q5.
REQUIREMENTS:
- All questions must be answerable from the reference material
- All 4 options should be plausible
- Correct answer must be clearly supported by material
- Keep explanations brief (1-2 sentences)
Generate 5 MCQs now:"""
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Class-12 teacher who creates high-quality MCQs from textbook content. You always follow the exact format specified."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1500,
            top_p=0.9
        )
        
        result = chat_completion.choices[0].message.content.strip()
        result = clean_mcq_output(result)
        
        cache_mcq(cache_key, result)
        
        print("✓ MCQs generated successfully")
        return result
        
    except Exception as e:
        error_msg = f"""Error calling Groq API: {str(e)}
Possible causes:
1. Rate limit exceeded (wait a moment)
2. Invalid API key
3. Network issue
Please try again in a few seconds."""
        print(f"❌ Groq API Error: {e}")
        return error_msg

def clean_mcq_output(text):
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        if (re.match(r'^Q\d+\.', line) or 
            line.startswith(('A)', 'B)', 'C)', 'D)', 'Answer:', 'Correct Answer:')) or
            not line):
            
            if line.startswith('Correct Answer:'):
                line = line.replace('Correct Answer:', 'Answer:')
            
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# ------------------------------
# HTML UI
# ------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Class 12 PCB MCQ Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .content { padding: 40px; }
        .form-group { margin-bottom: 25px; }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
            font-size: 16px;
        }
        select, input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            transition: border-color 0.3s;
        }
        select:focus, input:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            display: none;
        }
        .result.show { display: block; }
        .result h3 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.4em;
        }
        .mcq-content {
            background: white;
            padding: 25px;
            border-radius: 8px;
            white-space: pre-wrap;
            line-height: 1.9;
            font-size: 15px;
        }
        .loading {
            text-align: center;
            padding: 30px;
            display: none;
        }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .subject-tag {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            margin-right: 10px;
        }
        .bio { background: #d4edda; color: #155724; }
        .chem { background: #d1ecf1; color: #0c5460; }
        .phy { background: #f8d7da; color: #721c24; }
        .api-badge {
            background: #17a2b8;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Class 12 PCB MCQ Generator</h1>
            <p style="font-size: 1.1em; margin-bottom: 15px;">
                Generate practice MCQs from your textbooks 
                <span class="api-badge">⚡ Llama 3.3 70B</span>
            </p>
            <div>
                <span class="subject-tag bio">Biology</span>
                <span class="subject-tag chem">Chemistry</span>
                <span class="subject-tag phy">Physics</span>
            </div>
        </div>
        
        <div class="content">
            <div class="form-group">
                <label for="subject">📚 Select Subject</label>
                <select id="subject">
                    <option value="biology">Biology</option>
                    <option value="chemistry">Chemistry</option>
                    <option value="physics">Physics</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="topic">✏️ Enter Topic</label>
                <input type="text" id="topic" placeholder="e.g., Mitochondria, Chemical Bonding, Newton's Laws">
            </div>
            
            <button onclick="generateMCQs()">🚀 Generate 5 MCQs</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="color: #666; font-size: 16px;">Generating MCQs with AI...</p>
                <p style="color: #999; font-size: 13px; margin-top: 10px;">⚡ Usually takes 5-10 seconds</p>
            </div>
            
            <div class="result" id="result">
                <h3>📝 Generated MCQs:</h3>
                <div style="background: #d4edda; padding: 12px; border-radius: 6px; margin-bottom: 15px; color: #155724; font-size: 14px;">
                    ✓ <strong>High Quality:</strong> Generated by Llama 3.3 70B via Groq API
                </div>
                <div class="mcq-content" id="mcqContent"></div>
            </div>
        </div>
    </div>
    <script>
        async function generateMCQs() {
            const subject = document.getElementById('subject').value;
            const topic = document.getElementById('topic').value.trim();
            
            if (!topic) {
                alert('⚠️ Please enter a topic!');
                return;
            }
            
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const btn = document.querySelector('button');
            
            loading.classList.add('show');
            result.classList.remove('show');
            btn.disabled = true;
            btn.textContent = '⏳ Generating...';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({subject, topic})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('❌ Error: ' + data.error);
                    return;
                }
                
                document.getElementById('mcqContent').textContent = data.mcqs;
                result.classList.add('show');
            } catch (error) {
                alert('❌ Error: ' + error.message);
            } finally {
                loading.classList.remove('show');
                btn.disabled = false;
                btn.textContent = '🚀 Generate 5 MCQs';
            }
        }
        
        document.getElementById('topic').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                generateMCQs();
            }
        });
    </script>
</body>
</html>
"""

# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        subject = data.get("subject", "").lower()
        topic = data.get("topic", "")
        
        if not topic:
            return jsonify({"error": "Topic is required"}), 400
        
        if subject not in SUBJECTS:
            return jsonify({"error": "Invalid subject"}), 400
        
        print(f"\n🔍 Searching {subject} for: {topic}")
        
        context = rag_search(topic, subject, k=5)
        
        if not context or len(context.strip()) < 50:
            return jsonify({"error": f"No content found for: {topic}"}), 404
        
        print(f"✓ Context found ({len(context)} chars)")
        
        mcqs = generate_mcqs(context, topic, subject)
        
        return jsonify({"mcqs": mcqs, "subject": subject})
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "groq_available": groq_client is not None,
        "cache_size": len(MCQ_CACHE)
    })

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n🚀 Starting server on port {port}...\n")
    app.run(host="0.0.0.0", port=port, debug=False)
