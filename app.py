import pickle
import faiss
from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import torch
import os

app = Flask(__name__)

print("=" * 50)
print("Loading models and data...")
print("=" * 50)

# ------------------------------
# Load embedding model (CPU)
# ------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ Embedding model loaded")

# ------------------------------
# Download files from Hugging Face
# ------------------------------
REPO_ID = "Redfire-1234/pcb_tutor"

print("Downloading subject files from Hugging Face...")

# Download Biology files
bio_chunks_path = hf_hub_download(repo_id=REPO_ID, filename="bio_chunks.pkl", repo_type="model")
faiss_bio_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_bio.bin", repo_type="model")

# Download Chemistry files
chem_chunks_path = hf_hub_download(repo_id=REPO_ID, filename="chem_chunks.pkl", repo_type="model")
faiss_chem_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_chem.bin", repo_type="model")

# Download Physics files
phy_chunks_path = hf_hub_download(repo_id=REPO_ID, filename="phy_chunks.pkl", repo_type="model")
faiss_phy_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_phy.bin", repo_type="model")

# Load all subjects into memory
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

print(f"✓ Biology: {len(SUBJECTS['biology']['chunks'])} chunks loaded")
print(f"✓ Chemistry: {len(SUBJECTS['chemistry']['chunks'])} chunks loaded")
print(f"✓ Physics: {len(SUBJECTS['physics']['chunks'])} chunks loaded")

# ------------------------------
# Load LLM model (CPU)
# ------------------------------
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading LLM: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
).to(device)
print(f"✓ LLM loaded on {device}")

print("=" * 50)
print("All models loaded successfully!")
print("=" * 50)

# ------------------------------
# RAG Search in specific subject
# ------------------------------
def rag_search(query, subject, k=5):
    if subject not in SUBJECTS:
        return None
    
    chunks = SUBJECTS[subject]["chunks"]
    index = SUBJECTS[subject]["index"]
    
    q_emb = embed_model.encode([query]).astype("float32")
    D, I = index.search(q_emb, k)
    
    # Get the actual chunks
    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    
    return "\n\n".join(results)

# ------------------------------
# MCQ Generation with Step-by-Step Answer Selection
# ------------------------------
def generate_mcqs(context, topic, subject):
    prompt = f"""You are a Class-12 {subject.title()} teacher creating MCQs for students.
Topic: "{topic}"
Context from textbook:
{context}
TASK: Generate exactly 5 MCQs. For each MCQ:
1. Write a clear question
2. Create 4 options (A, B, C, D)
3. THINK CAREFULLY: Which option is correct according to the context?
4. Mark the correct answer
FORMAT (follow exactly):
Q1. [Your question here]
A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option]
Correct Answer: [Letter] - [Brief reason why this is correct]
IMPORTANT RULES:
✓ The correct answer MUST be supported by the context above
✓ Read all options carefully before selecting
✓ Do not guess - verify each answer from the context
✓ Make distractors (wrong options) realistic but clearly incorrect
Generate 5 MCQs now:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=900,
        temperature=0.15,  # Very low temperature for accuracy
        top_p=0.8,
        do_sample=True,
        repetition_penalty=1.15
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated MCQs
    if "Generate 5 MCQs now:" in result:
        result = result.split("Generate 5 MCQs now:")[-1].strip()
    
    return result

def verify_and_correct_answers(mcqs_text, context):
    """
    This function is kept for future enhancements
    """
    return mcqs_text

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
        .form-group {
            margin-bottom: 25px;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Class 12 PCB MCQ Generator</h1>
            <p style="font-size: 1.1em; margin-bottom: 15px;">Generate practice MCQs from your textbooks</p>
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
                <p style="color: #666; font-size: 16px;">Generating MCQs... This may take 30-60 seconds</p>
            </div>
            
            <div class="result" id="result">
                <h3>📝 Generated MCQs:</h3>
                <div style="background: #fff3cd; padding: 12px; border-radius: 6px; margin-bottom: 15px; color: #856404; font-size: 14px;">
                    ⚠️ <strong>Note:</strong> AI-generated answers may occasionally be incorrect. Please verify answers using your textbook.
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
        
        // Allow Enter key to submit
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
            return jsonify({"error": "Invalid subject. Choose biology, chemistry, or physics."}), 400
        
        print(f"\n🔍 Searching {subject} for topic: {topic}")
        
        # Retrieve context from RAG
        context = rag_search(topic, subject, k=5)
        
        if not context or len(context.strip()) < 50:
            return jsonify({"error": f"No relevant content found in {subject} for topic: {topic}"}), 404
        
        print(f"✓ Found context ({len(context)} chars)")
        
        # Generate MCQs
        print("🤖 Generating MCQs...")
        mcqs = generate_mcqs(context, topic, subject)
        
        print("✓ MCQs generated successfully")
        
        return jsonify({"mcqs": mcqs, "subject": subject})
    
    except Exception as e:
        print(f"❌ Error in /generate: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "subjects": {
            "biology": len(SUBJECTS["biology"]["chunks"]),
            "chemistry": len(SUBJECTS["chemistry"]["chunks"]),
            "physics": len(SUBJECTS["physics"]["chunks"])
        }
    })

# ------------------------------
# Run the App
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n🚀 Starting Flask on 0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)