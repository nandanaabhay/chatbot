from flask import Flask, render_template, request, Response, session, jsonify, redirect, url_for

from datetime import timedelta

import ollama

import os

from flask_sqlalchemy import SQLAlchemy



# ==========================

# INITIALIZATION & DIRECTORY FIX

# ==========================



app = Flask(__name__)

app.secret_key = "change-this-to-a-random-secret-key"

app.permanent_session_lifetime = timedelta(days=30)



# Ensure folders exist on your Mac before app startup

os.makedirs(os.path.join(app.root_path, "instance"), exist_ok=True)

os.makedirs(os.path.join(app.root_path, "uploads"), exist_ok=True)



# Absolute path tracking for SQLite database

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(app.root_path, 'instance', 'users.db')}"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False



db = SQLAlchemy(app)



# Import your friend's advanced RAG database functions

from rag.ingest import ingest_pdf

from rag.retriever import retrieve_docs



# ==========================

# DATABASE MODELS

# ==========================



class User(db.Model):
    __tablename__ = 'users'  # Force the table name to be 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    status = db.Column(db.Integer, default=1) # Add this column


class PDFContent(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, nullable=True)  # Links the uploaded document to a specific user account

    filename = db.Column(db.String(200), nullable=False)

    content = db.Column(db.Text, nullable=False)



class ChatHistory(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, nullable=True)

    question = db.Column(db.Text, nullable=False)

    answer = db.Column(db.Text, nullable=False)





# ==========================

# ROUTING CONTROLLERS

# ==========================



@app.route("/", methods=["GET"])

def home():

    if "user_id" in session:

        # Load database history only for authenticated users

        history = ChatHistory.query.filter_by(user_id=session["user_id"]).all()

    else:

        # Guests see a clean chat window

        history = []

        

    return render_template("index.html", chat_history=history)




# new user register
@app.route("/register", methods=["GET", "POST"])

def register():

    if request.method == "POST":

        username = request.form.get("username")

        email = request.form.get("email")

        password = request.form.get("password")



        existing_user = User.query.filter_by(email=email).first()

        if existing_user:

            return "Email already registered. <a href='/login'>Login here</a>"



        # pyrefly: ignore [unexpected-keyword]
        user = User(username=username, email=email, password=password)

        db.session.add(user)

        db.session.commit()

        return "Registration Successful! <a href='/login'>Click here to login</a>"



    return render_template("register.html")



# login

@app.route("/login", methods=["GET", "POST"])

def login():

    if request.method == "POST":

        email = request.form.get("email")

        password = request.form.get("password")



        user = User.query.filter_by(email=email).first()

        

        if not user:

            return redirect(url_for("register"))



        if user.password != password:

            return "Incorrect Password. Please try again."



        session.permanent = True

        session["user_id"] = user.id

        session["username"] = user.username

        return redirect(url_for("home"))



    return render_template("login.html")



# logout

from sqlalchemy import text # Ensure this is at the top of your file

@app.route('/logout')
def logout():
    user_id = session.get('user_id')
    
    # 1. Update status in database if user is logged in
    if user_id:
        db.session.execute(text("UPDATE users SET status = 0 WHERE id = :uid"), {"uid": user_id})
        db.session.commit()
    
    # 2. Clear the server-side session
    session.clear()
    
    # 3. Create the redirect response
    response = redirect(url_for("login"))
    
    # 4. Force browser to ignore cache so data doesn't persist after logout
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response





# ==========================

# DOCUMENT MANAGEMENT (PERMANENT STORAGE)

# ==========================



@app.route("/upload", methods=["POST"])

def upload():

    # Only logged-in profile users can assign documents to their account view

    if "user_id" not in session:

        return "Please log in to upload files to your account.", 401



    if "pdf" not in request.files:

        return "No file part", 400



    file = request.files["pdf"]

    if file.filename == "":

        return "No selected file", 400



    filepath = os.path.join("uploads", file.filename)

    file.save(filepath)



    current_user_id = session.get("user_id")



    try:

        # Check if this specific user has already uploaded this specific filename before

        existing_doc = PDFContent.query.filter_by(filename=file.filename, user_id=current_user_id).first()

        if not existing_doc:

            # pyrefly: ignore [unexpected-keyword]
            pdf = PDFContent(filename=file.filename, content="", user_id=current_user_id)   

            db.session.add(pdf)

            db.session.commit()



        print(f"\n[TERMINAL LOG] Ingesting {file.filename} into Chroma DB store...")

        ingest_pdf(filepath) 



        return f"File '{file.filename}' uploaded successfully!"

    except Exception as e:

        return f"Error parsing file: {str(e)}", 500





@app.route("/documents")

def documents():

    try:

        # Filter files strictly by who is currently logged into the session

        if "user_id" in session:

            docs = PDFContent.query.filter_by(user_id=session["user_id"]).all()

        else:

            docs = [] # Guests get a completely empty sidebar

            

        return jsonify([{"filename": doc.filename} for doc in docs])

    except Exception as e:

        return jsonify([])





# ==========================

# CHAT LOGIC & INDIVIDUAL DELETION

# ==========================



@app.route("/delete_message/<int:message_id>", methods=["POST"])

def delete_message(message_id):

    if "user_id" not in session:

        return "Unauthorized", 401

    

    try:

        msg = ChatHistory.query.filter_by(id=message_id, user_id=session["user_id"]).first()

        if msg:

            db.session.delete(msg)

            db.session.commit()

            return jsonify({"status": "success"})

        return jsonify({"status": "error"}), 404

    except Exception as e:

        return jsonify({"status": "error"}), 500





@app.route("/clear_history", methods=["POST"])

def clear_history():

    # Protection: Check if a user is actively logged in first

    if "user_id" not in session:

        return jsonify({"status": "unauthorized"}), 401

    

    try:

        current_user = session["user_id"]

        

        # 1. Wipe chat history from database

        ChatHistory.query.filter_by(user_id=current_user).delete()

        

        # 2. 🎯 ADD THIS: Wipe PDF records from database sidebar view

        PDFContent.query.filter_by(user_id=current_user).delete()

        

        db.session.commit()

        

        print(f"[DATABASE LOG] Wiped both chat history and PDF records for user ID: {current_user}")

        return jsonify({"status": "success"}), 200

        

    except Exception as e:

        db.session.rollback()

        print(f"[ERROR] Failed to clear history/documents: {str(e)}")

        return jsonify({"status": "error"}), 500





@app.route("/stream")

def stream():
    prompt = request.args.get("message", "")
    current_user_id = session.get("user_id")
    context_blocks = []
    
    print(f"\n[TERMINAL LOG] Querying hybrid search database for: '{prompt}'")
    retrieved_documents = retrieve_docs(prompt)
    
    for doc in retrieved_documents:
        text_content = getattr(doc, 'page_content', getattr(doc, 'text', str(doc)))
        source_file = getattr(doc, 'metadata', {}).get('filename', 'Uploaded Document')
        block = f"--- START SOURCE: {source_file} ---\n{text_content}\n--- END SOURCE: {source_file} ---"
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    # Define a strict system prompt that enforces grounding
    system_prompt = (
        "You are an advanced corporate PDF RAG Assistant. "
        "You must answer questions ONLY based on the provided 'Context' below. "
        "If the answer cannot be found in the provided context, or if the context is empty, "
        "you must reply with: 'I am sorry, but I do not have enough information in your uploaded documents to answer that question.' "
        "Do not use your own internal knowledge to provide an answer."
    )

    def generate():
        response_stream = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "system",
                    "content": f"{system_prompt}\n\nContext:\n{context}"
                },
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        
        
        # ... (rest of your generation code)



        full_answer = ""

        for response_chunk in response_stream:

            content = response_chunk["message"]["content"]

            full_answer += content

            yield content



        if current_user_id is not None:

            with app.app_context():

                # pyrefly: ignore [unexpected-keyword]
                history = ChatHistory(user_id=current_user_id, question=prompt, answer=full_answer)

                db.session.add(history)

                db.session.commit()

                print("[DATABASE LOG] Saved history log to user profile.")

        else:

            print("[DATABASE LOG] Guest query processed. Skipping history database dump.")



    return Response(generate(), mimetype="text/plain")





if __name__ == "__main__":

    with app.app_context():

        db.create_all()

    app.run(debug=True, port=5002)