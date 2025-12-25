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
# Chapter Names (Actual Textbook Chapters)
# ------------------------------
CHAPTER_NAMES = {
    "biology": [
        "Reproduction in Lower and Higher Plants",
        "Reproduction in Lower and Higher Animals",
        "Inheritance and Variation",
        "Molecular Basis of Inheritance",
        "Origin and Evolution of Life",
        "Plant Water Relation",
        "Plant Growth and Mineral Nutrition",
        "Respiration and Circulation",
        "Control and Co-ordination",
        "Human Health and Diseases",
        "Enhancement of Food Production",
        "Biotechnology",
        "Organisms and Populations",
        "Ecosystems and Energy Flow",
        "Biodiversity, Conservation and Environmental Issues"
    ],
    "chemistry": [
        "Solid State",
        "Solutions",
        "Ionic Equilibria",
        "Chemical Thermodynamics",
        "Electrochemistry",
        "Chemical Kinetics",
        "Elements of Groups 16, 17 and 18",
        "Transition and Inner transition Elements",
        "Coordination Compounds",
        "Halogen Derivatives",
        "Alcohols, Phenols and Ethers",
        "Aldehydes, Ketones and Carboxylic acids",
        "Amines",
        "Biomolecules",
        "Introduction to Polymer Chemistry",
        "Green Chemistry and Nanochemistry"
    ],
    "physics": [
        "Rotational Dynamics",
        "Mechanical Properties of Fluids",
        "Kinetic Theory of Gases and Radiation",
        "Thermodynamics",
        "Oscillations",
        "Superposition of Waves",
        "Wave Optics",
        "Electrostatics",
        "Current Electricity",
        "Magnetic Fields due to Electric Current",
        "Magnetic Materials",
        "Electromagnetic induction",
        "AC Circuits",
        "Dual Nature of Radiation and Matter",
        "Structure of Atoms and Nuclei",
        "Semiconductor Devices"
    ]
}

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
# Topic Validation (Check if topic belongs to subject)
# ------------------------------
def validate_topic_subject(topic, subject):
    """
    Validate if the topic belongs to the selected subject using LLM
    Returns True if valid, False otherwise
    """
    if not groq_client:
        return True  # Skip validation if API not available
    
    validation_prompt = f"""You are a Class 12 PCB subject expert. Determine if the following topic belongs to {subject.title()}.
Topic: "{topic}"
Subject: {subject.title()}
Class 12 {subject.title()} covers:
{"- Reproduction, Genetics, Evolution, Plant Physiology, Human Systems, Ecology, Biotechnology" if subject == "biology" else ""}
{"- Solid State, Solutions, Thermodynamics, Electrochemistry, Organic Chemistry, Coordination Compounds" if subject == "chemistry" else ""}
{"- Rotational Dynamics, Fluids, Thermodynamics, Waves, Optics, Electromagnetism, Modern Physics, Semiconductors" if subject == "physics" else ""}
Answer ONLY with "YES" if the topic belongs to {subject.title()}, or "NO" if it belongs to a different subject.
Answer:"""
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert at identifying which subject a topic belongs to. Answer only YES or NO."
                },
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        if "YES" in result:
            print(f"✓ Topic '{topic}' validated for {subject}")
            return True
        else:
            print(f"❌ Topic '{topic}' does NOT belong to {subject}")
            return False
            
    except Exception as e:
        print(f"⚠️ Validation failed: {e}")
        return True  # Allow on error to avoid blocking

# ------------------------------
# Chapter Detection (Using Actual Chapter Names)
# ------------------------------
def detect_chapter_from_list(context, topic, subject):
    """
    Detect chapter using the actual chapter list by matching keywords
    Returns None if topic doesn't match the subject
    """
    if subject not in CHAPTER_NAMES:
        return None
    
    chapters = CHAPTER_NAMES[subject]
    combined_text = (topic + " " + context[:1000]).lower()
    
    # Score each chapter based on keyword matching
    scores = {}
    for chapter in chapters:
        score = 0
        chapter_words = chapter.lower().split()
        
        # Check if chapter words appear in the content
        for word in chapter_words:
            if len(word) > 3:  # Ignore small words like "and", "the"
                if word in combined_text:
                    score += 1
        
        # Bonus if topic is similar to chapter name
        topic_words = topic.lower().split()
        for t_word in topic_words:
            if len(t_word) > 3 and t_word in chapter.lower():
                score += 2
        
        if score > 0:
            scores[chapter] = score
    
    # Return chapter with highest score
    if scores:
        best_chapter = max(scores.items(), key=lambda x: x[1])[0]
        print(f"✓ Matched chapter: {best_chapter} (score: {scores[best_chapter]})")
        return best_chapter
    
    # Fallback: Use LLM to choose from the list
    return detect_chapter_with_llm(context, topic, subject, chapters)

def detect_chapter_with_llm(context, topic, subject, chapters):
    """
    Use LLM to pick the correct chapter from the provided list
    Also verifies if the topic belongs to the subject
    """
    if not groq_client:
        return None
    
    chapter_list = "\n".join([f"{i+1}. {ch}" for i, ch in enumerate(chapters)])
    
    detection_prompt = f"""Based on the following textbook content and topic, identify which chapter from the Class 12 {subject.title()} textbook this content belongs to.
Topic: {topic}
Content snippet:
{context[:600]}
Available {subject.title()} chapters:
{chapter_list}
IMPORTANT: If the topic and content do NOT belong to {subject.title()}, respond with "NOT_MATCHING".
If it matches, respond with ONLY the chapter number and name exactly as listed (e.g., "5. Origin and Evolution of Life").
Response:"""
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert at identifying which chapter textbook content belongs to. You can recognize when content doesn't match the subject. If the topic is from a different subject than {subject.title()}, respond with 'NOT_MATCHING'."
                },
                {
                    "role": "user",
                    "content": detection_prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        
        # Check if topic doesn't match the subject
        if "NOT_MATCHING" in result.upper() or "NOT MATCHING" in result.upper():
            print(f"⚠️ Topic '{topic}' doesn't belong to {subject}")
            return None
        
        # Extract chapter name from response (remove number prefix if present)
        chapter = re.sub(r'^\d+\.\s*', '', result).strip()
        
        # Verify it's in our list
        for ch in chapters:
            if ch.lower() in chapter.lower() or chapter.lower() in ch.lower():
                print(f"✓ LLM detected chapter: {ch}")
                return ch
        
        print(f"⚠️ LLM response not in list: {result}")
        return None
        
    except Exception as e:
        print(f"⚠️ Chapter detection failed: {e}")
        return None

# ------------------------------
# MCQ Generation
# ------------------------------
def generate_mcqs(context, topic, subject, num_questions=5):
    # Check if Groq is available
    if not groq_client:
        error_msg = """ERROR: Groq API not initialized!
Please check:
1. GROQ_API_KEY is set in Space Settings → Repository secrets
2. API key is valid (get one from https://console.groq.com/keys)
3. Space has been restarted after adding the key
Current status: API key not found or invalid."""
        return error_msg, None
    
    # Check cache
    context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
    cache_key = get_cache_key(topic, subject, context_hash) + f":{num_questions}"
    
    if cache_key in MCQ_CACHE:
        print("✓ Using cached MCQs")
        return MCQ_CACHE[cache_key]["mcqs"], MCQ_CACHE[cache_key]["chapter"]
    
    print(f"🤖 Generating {num_questions} MCQs for {subject} - {topic}")
    
    # Detect the chapter from our actual chapter list
    chapter = detect_chapter_from_list(context, topic, subject)
    
    # If chapter is None, topic doesn't belong to this subject
    if chapter is None:
        error_msg = f"❌ The topic '{topic}' does not belong to {subject.title()}.\n\nPlease enter a topic related to {subject.title()} or select the correct subject."
        print(f"⚠️ Topic mismatch: '{topic}' not in {subject}")
        return error_msg, None
    
    prompt = f"""You are a Class-12 {subject.title()} teacher creating MCQs.
Topic: "{topic}"
Chapter: "{chapter}"
Reference material from textbook:
{context[:1500]}
Generate exactly {num_questions} multiple-choice questions based on the reference material.
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
Continue for Q3, Q4, Q5{"..." if num_questions > 5 else ""}.
REQUIREMENTS:
- All questions must be answerable from the reference material
- All 4 options should be plausible
- Correct answer must be clearly supported by material
- Keep explanations brief (1-2 sentences)
Generate {num_questions} MCQs now:"""
    
    try:
        # Adjust max_tokens based on number of questions
        max_tokens = min(3000, 300 * num_questions)
        
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
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        result = chat_completion.choices[0].message.content.strip()
        result = clean_mcq_output(result)
        
        # Cache both MCQs and chapter
        cache_mcq(cache_key, {"mcqs": result, "chapter": chapter})
        
        print("✓ MCQs generated successfully")
        return result, chapter
        
    except Exception as e:
        error_msg = f"""Error calling Groq API: {str(e)}
Possible causes:
1. Rate limit exceeded (wait a moment)
2. Invalid API key
3. Network issue
Please try again in a few seconds."""
        print(f"❌ Groq API Error: {e}")
        return error_msg, chapter

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
        .form-row {
            display: flex;
            gap: 15px;
            margin-bottom: 25px;
        }
        .form-row .form-group {
            margin-bottom: 0;
        }
        .form-row .form-group:first-child {
            flex: 2;
        }
        .form-row .form-group:last-child {
            flex: 0 0 120px;
        }
        .form-row .form-group:last-child select {
            padding: 15px 10px;
            font-size: 15px;
        }
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
        .chapter-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 16px;
        }
        .chapter-icon {
            font-size: 24px;
        }
        .chapter-text {
            flex: 1;
        }
        .chapter-name {
            font-weight: 700;
            font-size: 18px;
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
            
            <div class="form-row">
                <div class="form-group">
                    <label for="topic">✏️ Enter Topic</label>
                    <input type="text" id="topic" placeholder="e.g., Mitochondria, Chemical Bonding, Newton's Laws">
                </div>
                
                <div class="form-group">
                    <label for="numQuestions">🔢 MCQs</label>
                    <select id="numQuestions">
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5" selected>5</option>
                        <option value="6">6</option>
                        <option value="7">7</option>
                        <option value="8">8</option>
                        <option value="9">9</option>
                        <option value="10">10</option>
                        <option value="11">11</option>
                        <option value="12">12</option>
                        <option value="13">13</option>
                        <option value="14">14</option>
                        <option value="15">15</option>
                        <option value="16">16</option>
                        <option value="17">17</option>
                        <option value="18">18</option>
                        <option value="19">19</option>
                        <option value="20">20</option>
                    </select>
                </div>
            </div>
            
            <button onclick="generateMCQs()">🚀 Generate MCQs</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="color: #666; font-size: 16px;">Generating MCQs with AI...</p>
                <p style="color: #999; font-size: 13px; margin-top: 10px;">⚡ Detecting chapter from textbook...</p>
            </div>
            
            <div class="result" id="result">
                <h3>📝 Generated MCQs:</h3>
                
                <div class="chapter-info" id="chapterInfo" style="display: none;">
                    <span class="chapter-icon">📖</span>
                    <div class="chapter-text">
                        <div style="font-size: 13px; opacity: 0.9;">Chapter:</div>
                        <div class="chapter-name" id="chapterName"></div>
                    </div>
                </div>
                
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
            const numQuestions = parseInt(document.getElementById('numQuestions').value);
            
            if (!topic) {
                alert('⚠️ Please enter a topic!');
                return;
            }
            
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const btn = document.querySelector('button');
            const chapterInfo = document.getElementById('chapterInfo');
            
            loading.classList.add('show');
            result.classList.remove('show');
            chapterInfo.style.display = 'none';
            btn.disabled = true;
            btn.textContent = '⏳ Generating...';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({subject, topic, num_questions: numQuestions})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('❌ Error: ' + data.error);
                    return;
                }
                
                // Display chapter info
                if (data.chapter && data.chapter !== 'Unknown Chapter') {
                    document.getElementById('chapterName').textContent = data.chapter;
                    chapterInfo.style.display = 'flex';
                }
                
                document.getElementById('mcqContent').textContent = data.mcqs;
                result.classList.add('show');
            } catch (error) {
                alert('❌ Error: ' + error.message);
            } finally {
                loading.classList.remove('show');
                btn.disabled = false;
                btn.textContent = '🚀 Generate MCQs';
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
        num_questions = data.get("num_questions", 5)
        
        # Validate num_questions
        try:
            num_questions = int(num_questions)
            if num_questions < 1 or num_questions > 20:
                num_questions = 5
        except:
            num_questions = 5
        
        if not topic:
            return jsonify({"error": "Topic is required"}), 400
        
        if subject not in SUBJECTS:
            return jsonify({"error": "Invalid subject"}), 400
        
        print(f"\n🔍 Validating topic for {subject}...")
        
        # STEP 1: Validate if topic belongs to subject (BEFORE RAG search)
        if not validate_topic_subject(topic, subject):
            subject_names = {
                "biology": "Biology",
                "chemistry": "Chemistry", 
                "physics": "Physics"
            }
            error_msg = f"The topic '{topic}' does not appear to be related to {subject_names[subject]}.\n\nPlease either:\n• Enter a {subject_names[subject]}-related topic, or\n• Select the correct subject for this topic"
            return jsonify({"error": error_msg}), 400
        
        print(f"✓ Topic validated for {subject}")
        print(f"🔍 Searching {subject} for: {topic}")
        
        # STEP 2: RAG search
        context = rag_search(topic, subject, k=5)
        
        if not context or len(context.strip()) < 50:
            return jsonify({"error": f"No content found for: {topic}"}), 404
        
        print(f"✓ Context found ({len(context)} chars)")
        
        # STEP 3: Generate MCQs
        mcqs, chapter = generate_mcqs(context, topic, subject, num_questions)
        
        # Check if there was a subject mismatch
        if chapter is None:
            return jsonify({"error": mcqs}), 400
        
        return jsonify({
            "mcqs": mcqs, 
            "subject": subject,
            "chapter": chapter
        })
    
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
