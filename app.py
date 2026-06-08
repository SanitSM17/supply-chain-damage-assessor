import streamlit as st
import google.generativeai as genai
import chromadb
from PIL import Image
import json
import pandas as pd
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==============================================================================
# 1. UI & STYLING CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Supply Chain Damage Assessor PRO", page_icon="📦", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 2.3rem; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 1.1rem; color: #4B5563; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">📦 Industrial Supply-Chain Damage Assessor (Pro Edition)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Multimodal Vision, Structured Data Tables, Local Vector Memory, and Automated Escalations.</p>', unsafe_allow_html=True)

# ==============================================================================
# 2. API KEYS & CREDENTIALS VERIFICATION
# ==============================================================================
# Tries to pull keys from Streamlit Cloud Secrets, falls back to the sidebar for testing
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("🔑 Enter Gemini API Key:", type="password", help="Get a free key from Google AI Studio")

st.sidebar.markdown("---")
st.sidebar.header("✉️ Email Escalation Setup (Optional)")
sender_email = st.sidebar.text_input("Sender Gmail", placeholder="bot@gmail.com")
sender_password = st.sidebar.text_input("Gmail App Password", type="password", placeholder="xxxx xxxx xxxx xxxx")
receiver_email = st.sidebar.text_input("Manager Destination Email", placeholder="manager@company.com")

# ==============================================================================
# 3. HELPER FUNCTIONS (DATABASES & EMAIL OUTBOUND)
# ==============================================================================
@st.cache_resource
def initialize_vector_db():
    """Initializes local, zero-cost ChromaDB tracking folder within Streamlit disk space."""
    client = chromadb.PersistentClient(path="./chroma_db_storage")
    collection = client.get_or_create_collection(name="damage_history_ledger")
    return collection

def trigger_alert_email(con_id, carrier, details, severity):
    """Fires a completely free notification email using native Python and Google SMTP."""
    if not (sender_email and sender_password and receiver_email):
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"🚨 {severity.upper()} SUPPLY CHAIN ALERT: Consignment {con_id}"
        
        body = f"""
        Warning: A priority incident requires immediate logistics review.
        
        - Consignment / Waybill ID: {con_id}
        - Shipping Carrier: {carrier}
        - Risk Severity: {severity}
        - System Timestamp: {datetime.date.today().strftime('%Y-%m-%d')}
        
        AI Assessment & Ingestion Logs:
        {details}
        
        Action Required: Log into the Streamlit Quality Ledger dashboard to verify records.
        """
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"SMTP Notification Pipeline Failed: {e}")
        return False

# ==============================================================================
# 4. MAIN APPLICATION CORE WORKFLOWS
# ==============================================================================
if api_key:
    genai.configure(api_key=api_key)
    db_collection = initialize_vector_db()
    
    # Session data sidebars
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Current Cargo Metadata")
    consignment_id = st.sidebar.text_input("Consignment / Waybill ID", value="CON-2026-7719")
    carrier_name = st.sidebar.selectbox("Logistics Carrier", ["DHL Express", "FedEx Cargo", "Maersk", "Blue Dart"])
    
    # Navigation Tabs splits functionality cleanly
    tab1, tab2 = st.tabs(["🚀 Real-Time Scanner & Ledger", "🔍 Semantic Vector Memory"])
    
    with tab1:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("📸 Visual Evidence Entry")
            input_method = st.radio("Select Capture Mode:", ["Use Device Camera", "Upload Image File"])
            
            img_file = st.camera_input("Scan barcode/package damage") if input_method == "Use Device Camera" else st.file_uploader("Upload incident photo...", type=["jpg", "jpeg", "png"])
            
            if img_file and input_method == "Upload Image File":
                st.image(Image.open(img_file), caption="Uploaded Cargo Evidence Preview", use_container_width=True)
                
        with col2:
            st.subheader("🤖 AI Analysis & Structured Data Output")
            
            if img_file:
                if st.button("Run Multimodal Assessment", type="primary"):
                    with st.spinner("Extracting parameters and validating structural schemas..."):
                        try:
                            pil_image = Image.open(img_file)
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            
                            # Constructing prompt specifying strict JSON returns
                            prompt = f"""
                            You are an expert industrial quality control engine. Analyze this cargo damage photo.
                            You must respond exclusively with a valid JSON object matching the template below. 
                            Do not add background text, conversational conversational lines, or wrap it in standard markdown blocks.
                            
                            Schema Design: {{
                                "Consignment_ID": "{consignment_id}",
                                "Carrier": "{carrier_name}",
                                "Damage_Type": "Primary visual classification (e.g., Crushed Base, Liquid Tear, Pierced Structural Core)",
                                "Severity": "Low, Medium, High, or Critical",
                                "Estimated_Loss_USD": Range integer between 50 and 5000 based on damage depth,
                                "Operational_Action": "Quarantine, Repackage, or Discard",
                                "Detailed_Narrative": "A clean 2-sentence formal report detail outlining visible flaws for insurance brokers."
                            }}
                            """
                            
                            # Forcing native JSON formatting at the model configuration level
                            response = model.generate_content(
                                [prompt, pil_image],
                                generation_config={"response_mime_type": "application/json"}
                            )
                            
                            # Read JSON, transform to a tabular structure and store to session
                            parsed_json = json.loads(response.text)
                            st.session_state["active_record"] = pd.DataFrame([parsed_json])
                            st.success("Successfully parsed image into a database-ready schema row!")
                            
                        except Exception as e:
                            st.error(f"Multimodal Analysis Pipeline failed: {str(e)}")
                
                # Interactive CRUD Section if record exists
                if "active_record" in st.session_state:
                    st.markdown("#### 📝 Edit AI Extracted Ledger Data Live")
                    st.caption("Double-click any data cell below to fine-tune values manually before system commit:")
                    
                    # Convert static framework to live editable component spreadsheets!
                    edited_df = st.data_editor(st.session_state["active_record"], use_container_width=True, num_rows="fixed")
                    
                    if st.button("💾 Commit Record to Memory & ERP"):
                        with st.spinner("Processing background database pipelines..."):
                            try:
                                data_row = edited_df.to_dict(orient="records")[0]
                                narrative_summary = data_row["Detailed_Narrative"]
                                computed_severity = data_row["Severity"]
                                
                                # A. Generate vector search representation for historical auditing
                                vector_res = genai.embed_content(model="models/text-embedding-004", contents=narrative_summary)
                                vector_embedding = vector_res['embedding']
                                
                                # B. Upsert inside local Chroma DB instance folder
                                db_collection.upsert(
                                    ids=[data_row["Consignment_ID"]],
                                    embeddings=[vector_embedding],
                                    documents=[narrative_summary],
                                    metadatas=[{
                                        "id": data_row["Consignment_ID"],
                                        "carrier": data_row["Carrier"],
                                        "severity": computed_severity,
                                        "loss": int(data_row["Estimated_Loss_USD"])
                                    }]
                                )
                                st.success(f"Saved into local vector storage index!")
                                
                                # C. Core Automation Escalation Layer Rule
                                if computed_severity in ["High", "Critical"]:
                                    st.warning(f"⚠️ Severity marked as '{computed_severity}'. Activating email routing rules...")
                                    status = trigger_alert_email(
                                        data_row["Consignment_ID"], 
                                        data_row["Carrier"], 
                                        narrative_summary, 
                                        computed_severity
                                    )
                                    if status:
                                        st.success("Priority alert routed to operations manager email safely!")
                                    else:
                                        st.info("System notification bypassed. Add credentials in the sidebar to test.")
                                        
                                st.balloons()
                            except Exception as db_err:
                                st.error(f"Database/Alert routing failed: {str(db_err)}")
            else:
                st.info("💡 Scan a package via camera or drag-and-drop imagery to trigger active operations.")
                
    with tab2:
        st.subheader("🔍 Query Historical Workspace Trends")
        st.caption("Search past records contextually (e.g., 'crushed bottom pallets from wet weather conditions') instead of strict keyword text matches.")
        
        search_term = st.text_input("Enter natural language vector query...", placeholder="Type query parameter...")
        
        if search_term and st.button("Scan Corporate Memory Banks"):
            with st.spinner("Querying local context collections..."):
                # Embed the query
                query_vector = genai.embed_content(model="models/text-embedding-004", contents=search_term)['embedding']
                
                # Fetch closest matches
                matched_records = db_collection.query(query_embeddings=[query_vector], n_results=3)
                
                if matched_records['documents'] and len(matched_records['documents'][0]) > 0:
                    for idx in range(len(matched_records['documents'][0])):
                        doc_text = matched_records['documents'][0][idx]
                        metadata = matched_records['metadatas'][0][idx]
                        
                        with st.expander(f"📦 ID: {metadata['id']} | Carrier: {metadata['carrier']} | Priority: {metadata['severity']}"):
                            st.write(f"**Financial Impact:** ${metadata['loss']}")
                            st.info(f"**Logged Report Narrative:** {doc_text}")
                else:
                    st.warning("No contextually matching damage logs found inside the vector framework directories.")
else:
    st.warning("⚠️ Enter your Gemini API key in the sidebar workspace parameters to activate processing modules.")
