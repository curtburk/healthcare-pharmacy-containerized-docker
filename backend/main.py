from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
from llama_cpp import Llama
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical AI Drug Interaction Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "/app/frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Pydantic models ─────────────────────────────────────────────────────────

class SimpleDrugQuery(BaseModel):
    drug1: str
    drug2: str

class ComplexDrugQuery(BaseModel):
    medications: List[str]
    age: int
    conditions: List[str]
    lab_values: Optional[dict] = {}
    additional_context: Optional[str] = ""


# ── Model loading ────────────────────────────────────────────────────────────

model_path = os.environ.get("MODEL_PATH", "/app/models/medical-mixtral-q4.gguf")
logger.info(f"Loading quantized model from: {model_path}")

if not os.path.exists(model_path):
    logger.error(f"Model file not found: {model_path}")
    logger.error("Ensure the model is in the ./models/ directory on the host.")
    raise FileNotFoundError(f"Model not found: {model_path}")

llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,   # Use all layers on GPU
    n_ctx=2048,
    n_batch=512,
    n_threads=int(os.environ.get("OMP_NUM_THREADS", "8")),
    verbose=False
)

logger.info("✅ Quantized medical model loaded and ready!")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    try:
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found. Check container build.</h1>",
            status_code=500
        )

@app.post("/api/simple_interaction")
async def simple_drug_interaction(query: SimpleDrugQuery):
    try:
        prompt = f"""[INST] As a clinical pharmacist, analyze the drug interaction between {query.drug1} and {query.drug2}.
        
        Provide:
        1. Interaction severity (None/Low/Moderate/High/Contraindicated)
        2. Clinical mechanism of interaction
        3. Patient monitoring recommendations
        4. Alternative medications if interaction is significant
        
        Be concise but thorough. [/INST]"""

        start_time = time.time()

        output = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["</s>", "[INST]"]
        )

        response = output['choices'][0]['text'].strip()
        inference_time = time.time() - start_time

        return JSONResponse(content={
            "success": True,
            "drug1": query.drug1,
            "drug2": query.drug2,
            "analysis": response,
            "inference_time": round(inference_time, 2),
            "complexity_level": "simple"
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/complex_interaction")
async def complex_drug_interaction(query: ComplexDrugQuery):
    try:
        medications_str = ", ".join(query.medications)
        conditions_str = ", ".join(query.conditions) if query.conditions else "None reported"

        lab_context = ""
        if query.lab_values:
            lab_items = [f"{k}: {v}" for k, v in query.lab_values.items()]
            lab_context = f"Lab values: {', '.join(lab_items)}"

        prompt = f"""[INST] You are a clinical decision support AI. Analyze this complex patient case:

        Patient Profile:
        - Age: {query.age} years old
        - Current medications: {medications_str}
        - Medical conditions: {conditions_str}
        {lab_context}
        
        Provide analysis of:
        1. Drug-drug interactions (rank by severity)
        2. Risk assessment
        3. Monitoring recommendations
        4. Priority actions
        
        Be thorough but concise. [/INST]"""

        start_time = time.time()

        output = llm(
            prompt,
            max_tokens=400,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["</s>", "[INST]"]
        )

        response = output['choices'][0]['text'].strip()
        inference_time = time.time() - start_time

        risk_score = min(10, len(query.medications) * 1.5 + (query.age / 10) - 5)
        risk_level = "Low" if risk_score < 4 else "Moderate" if risk_score < 7 else "High"

        return JSONResponse(content={
            "success": True,
            "patient_summary": {
                "medications": query.medications,
                "age": query.age,
                "conditions": query.conditions
            },
            "analysis": response,
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "inference_time": round(inference_time, 2),
            "complexity_level": "complex"
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "Fine-tuned Mixtral Medical (Quantized Q4)",
        "inference_engine": "llama.cpp"
    }

@app.get("/api/sample_queries")
async def get_sample_queries():
    return {
        "simple_examples": [
            {"drug1": "warfarin", "drug2": "aspirin"},
            {"drug1": "metformin", "drug2": "contrast dye"},
            {"drug1": "simvastatin", "drug2": "clarithromycin"}
        ],
        "complex_examples": [
            {
                "description": "Elderly patient with polypharmacy",
                "medications": ["warfarin", "aspirin", "omeprazole", "metoprolol", "amlodipine"],
                "age": 78,
                "conditions": ["atrial fibrillation", "hypertension", "GERD"],
                "lab_values": {"INR": "3.2", "CrCl": "45 mL/min"}
            },
            {
                "description": "Diabetic patient with cardiovascular disease",
                "medications": ["metformin", "glipizide", "atorvastatin", "lisinopril", "clopidogrel"],
                "age": 65,
                "conditions": ["type 2 diabetes", "CAD s/p stent", "hyperlipidemia"],
                "lab_values": {"HbA1c": "8.2%", "eGFR": "58"}
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn

    host_ip = os.environ.get("HOST_IP", "")
    port = 8000

    print("\n" + "=" * 60)
    print("  Medical AI Drug Interaction Demo")
    print("  Fine-tuned Mixtral-8x7B | HP ZGX Nano")
    print("=" * 60)
    if host_ip:
        print(f"\n  ➜  http://{host_ip}:{port}")
    else:
        print(f"\n  ➜  http://localhost:{port}")
    print(f"  ➜  Health: http://{'localhost' if not host_ip else host_ip}:{port}/api/health")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
