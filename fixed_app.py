from flask import Flask, session, jsonify, render_template, request
from flask_session import Session
import os
import uuid

# ===== CORRECTED INITIALIZATION ORDER =====
app = Flask(__name__)

# 1. Basic Flask config FIRST
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key-' + str(uuid.uuid4())),
    DEBUG=False,
    SESSION_TYPE='filesystem',
    SESSION_FILE_DIR=os.path.abspath('./flask_session'),
    SESSION_PERMANENT=False,
    SESSION_COOKIE_NAME='eye_ai_session',
    SESSION_COOKIE_HTTPONLY=False,  # Allow debugging
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,  # HTTP for dev
    SESSION_COOKIE_DOMAIN=None,  # Auto-detect
    SESSION_COOKIE_PATH='/',
)

# 2. Initialize Flask-Session AFTER config
Session(app)

# 3. Ensure directories exist
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
os.makedirs('./static', exist_ok=True)

# ===== DEBUG MIDDLEWARE =====
@app.before_request
def log_request():
    print(f"\n{'='*50}")
    print(f"REQUEST: {request.method} {request.path}")
    print(f"PID: {os.getpid()}")
    print(f"APP ID: {id(app)}")
    print(f"SESSION ID: {getattr(session, 'sid', 'NO_SID')}")
    print(f"COOKIES: {dict(request.cookies)}")
    
    # Check session files
    session_files = os.listdir(app.config['SESSION_FILE_DIR'])
    print(f"SESSION FILES: {len(session_files)}")
    for f in session_files[:3]:  # Limit output
        print(f"  - {f}")
    print(f"SESSION KEYS: {list(session.keys())}")
    print(f"{'='*50}\n")

@app.after_request  
def log_response(response):
    if hasattr(response, 'headers'):
        set_cookie = response.headers.get('Set-Cookie')
        if set_cookie:
            print(f"RESPONSE SET-COOKIE: {set_cookie[:100]}...")
    return response

# ===== CLEANED ROUTES =====
@app.route("/")
def index():
    session['test'] = 'homepage_visited'
    session.modified = True
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify(error="No image"), 400
    
    # Clear but preserve test
    test_val = session.get('test')
    session.clear()
    if test_val:
        session['test'] = test_val
    
    # Save image
    file = request.files["image"]
    img_path = os.path.join("./static", "captured.jpg")
    file.save(img_path)
    
    # Store SIMPLE data only
    session['agent_outputs'] = {
        'diagnosis': 'Test diagnosis data',
        'validation': 'Test validation data', 
        'risk': 'Test risk data',
        'explanation': 'Test explanation',
        'report': 'Test report data'
    }
    
    session.modified = True
    print(f"UPLOAD: Stored {len(session)} keys in session")
    
    return jsonify(prediction=['N'])

@app.route("/agent/<agent_name>")
def agent(agent_name):
    outputs = session.get('agent_outputs', {})
    
    if not outputs:
        print(f"AGENT: No outputs found in session")
        return jsonify({
            'error': 'Please upload an image first',
            'session_keys': list(session.keys()),
            'session_id': getattr(session, 'sid', None)
        })
    
    if agent_name not in outputs:
        return jsonify(error=f'Invalid agent: {agent_name}')
    
    return jsonify({
        'agent': agent_name,
        'output': outputs[agent_name],
        'session_id': getattr(session, 'sid', None)
    })

@app.route("/debug/session")
def debug_session():
    return jsonify({
        'session_id': getattr(session, 'sid', None),
        'session_keys': list(session.keys()),
        'cookies': dict(request.cookies),
        'session_files': len(os.listdir(app.config['SESSION_FILE_DIR']))
    })

# ===== PRODUCTION-SAFE RUN =====
if __name__ == "__main__":
    print(f"Starting Flask app...")
    print(f"Session dir: {app.config['SESSION_FILE_DIR']}")
    print(f"Cookie name: {app.config['SESSION_COOKIE_NAME']}")
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        use_reloader=False
    )
