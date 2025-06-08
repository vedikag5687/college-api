from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import os
from datetime import datetime

app = FastAPI(
    title="College Recommendation API",
    description="API for JEE-based college recommendations",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class UserPreferences(BaseModel):
    name: str = Field(..., min_length=1, description="User's name")
    phone: str = Field(..., min_length=10, description="User's phone number")
    gender: str = Field(..., description="Gender preference")
    category: str = Field(..., description="Category (SC, ST, OBC, etc.)")
    state: str = Field(..., description="Home state")
    degrees: List[str] = Field(..., min_items=1, description="Preferred degrees")
    branches: List[str] = Field(..., min_items=1, description="Preferred branches")
    rank: int = Field(..., gt=0, description="JEE Mains rank")

class CollegeResult(BaseModel):
    college_name: str
    close_rank: int

class RecommendationResponse(BaseModel):
    success: bool
    message: str
    nits: List[CollegeResult]
    iiits: List[CollegeResult]
    nit_count: int
    iiit_count: int
    saved_file: Optional[str] = None

class OptionsResponse(BaseModel):
    genders: List[str]
    categories: List[str]
    states: List[str]
    degrees: List[str]
    branches: List[str]

SHEET_URL = "https://docs.google.com/spreadsheets/d/1LW-TpBjX1mK1JT-kraWZ5g5D6ERD_PszqG6qucVYE3s/edit"

# Data storage (will be loaded on startup)
sheets_data = {}

# Hardcoded options
GENDER_OPTIONS = ["Gender-Neutral", "Female-only (including Supernumerary)"]
CATEGORY_OPTIONS = ["SC", "ST", "EWS", "EWS (PwD)", "OBC-NCL", "OBC-NCL (PwD)", "OPEN", "OPEN (PwD)", "SC (PwD)", "ST (PwD)"]
STATE_OPTIONS = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Manipur", "Meghalaya", "Mizoram", "Maharashtra", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"]

DEGREE_OPTIONS = [
    "Bachelor of Technology",
    "Bachelor and Master of Technology (Dual Degree)",
    "Integrated Master of Science",
    "Integrated B. Tech. and M. Tech",
    "Bachelor of Architecture",
    "Bachelor of Planning",
    "Bachelor of Science and Master of Science (Dual Degree)",
    "B.Tech. + M.Tech./MS (Dual Degree)",
]

BRANCH_OPTIONS = [
    "Artificial Intelligence",
    "Aerospace Engineering",
    "Architecture",
    "Architecture and Planning",
    "Artificial Intelligence and Data Engineering",
    "Artificial Intelligence and Data Science",
    "Artificial Intelligence and Machine Learning",
    "B.Tech in Mathematics and Computing",
    "B.Tech in Mechanical Engineering and M.Tech in AI and Robotics",
    "B.Tech. in Electronics and Communication Engineering and M.Tech. in Communication Systems",
    "B.Tech. in Electronics and Communication Engineering and M.Tech. in VLSI Design",
    "Bio Medical Engineering",
    "Bio Technology",
    "Biotechnology",
    "Biotechnology and Biochemical Engineering",
    "Civil Engineering",
    "Ceramic Engineering",
    "Ceramic Engineering and M.Tech Industrial Ceramic",
    "Chemical Engineering",
    "Chemical Technology",
    "Chemistry",
    "Civil Engineering with Specialization in Construction Technology and Management",
    "Computational and Data Science",
    "Computational Mathematics",
    "Computer Science",
    "Computer Science and Artificial Intelligence",
    "Computer Science and Business",
    "Computer Science and Engineering",
    "Computer Science and Engineering (Artificial Intelligence)",
    "Computer Science Engineering (Artificial lntelligence and Machine Learning)",
    "Computer Science Engineering (Data Science and Analytics)",
    "Computer Science and Engineering (Cyber Physical System)",
    "Computer Science and Engineering (Cyber Security)",
    "Computer Science and Engineering (Data Science)",
    "Computer Science and Engineering (with Specialization of Data Science and Artificial Intelligence)",
    "Computer Science Engineering (Human Computer lnteraction and Gaming Technology)",
    "CSE ( Data Science & Analytics)",
    "Data Science and Engineering",
    "Data Science and Artificial Intelligence",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Electrical Engineering",
    "Electrical Engineering with Specialization In Power System Engineering",
    "Electronics and Communication Engineering (Internet of Things)",
    "Electronics and Communication Engineering (with Specialization of Embedded Systems and Internet of Things)",
    "Electronics and Communication Engineering with specialization in Design and Manufacturing",
    "Electronics and Communication Engineering (VLSI Design and Technology)",
    "Electronics and Communication Engineering with Specialization in Microelectronics and VLSI System Design",
    "Electronics and Communication Engineering with specialization in VLSI and Embedded Systems",
    "Electronics and Instrumentation Engineering",
    "Electronics and Telecommunication Engineering",
    "Electronics and VLSI Engineering",
    "Engineering and Computational Mechanics",
    "Engineering Physics",
    "Food Process Engineering",
    "Industrial and Production Engineering",
    "Information Technology-Business Informatics",
    "Integrated B. Tech.(IT) and M. Tech (IT)",
    "Integrated B. Tech.(IT) and MBA",
    "Industrial Chemistry",
    "Industrial Design",
    "Industrial Internet of Things",
    "Information Technology",
    "Instrumentation and Control Engineering",
    "Life Science",
    "Material Science and Engineering",
    "Materials Engineering",
    "Materials Science and Engineering",
    "Materials Science and Metallurgical Engineering",
    "Mathematics",
    "Mathematics & Computing",
    "Mathematics and Computing",
    "Mathematics and Computing Technology",
    "Mathematics and Scientific Computing",
    "Mathematics and Data Science",
    "Mechanical Engineering",
    "Mechanical Engineering with Specialization in Manufacturing and Industrial Engineering",
    "Mechanical Engineering with specialization in Design and Manufacturing",
    "Mechatronics and Automation Engineering",
    "Metallurgical and Materials Engineering",
    "Metallurgy and Materials Engineering",
    "Microelectronics & VLSI Engineering",
    "Mining Engineering",
    "Physics",
    "Planning",
    "Production and Industrial Engineering",
    "Production Engineering",
    "ROBOTICS & AUTOMATION",
    "SUSTAINABLE ENERGY TECHNOLOGIES",
    "Smart Manufacturing",
    "Textile Technology",
    "VLSI Design and Technology"
]

# Utility functions
def load_sheets_data():
    """Load data from Google Sheets"""
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Get credentials from environment variable
        credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        if not credentials_json:
            raise Exception("Google credentials not configured")
        
        credentials_dict = json.loads(credentials_json)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
        client = gspread.authorize(creds)
        
        spreadsheet = client.open_by_url(SHEET_URL)
        sheet_names = ["NITs Round 5", "IIITs Round 5", "IITs Round 5"]
        
        data = {}
        for name in sheet_names:
            worksheet = spreadsheet.worksheet(name)
            df = pd.DataFrame(worksheet.get_all_records())
            
            # Clean column names
            df.columns = (
                df.columns
                .str.encode('ascii', 'ignore').str.decode('ascii')
                .str.strip()
                .str.lower()
                .str.replace(r'\s+', ' ', regex=True)
            )
            
            # Clean "close rank"
            df['close rank'] = pd.to_numeric(df['close rank'], errors='coerce')
            data[name.lower()] = df.dropna(subset=['close rank'])
        
        return data
    except Exception as e:
        print(f"Error loading sheets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load sheets data: {str(e)}")

def filter_colleges(df, gender, category, rank, degrees, branches, state=None, is_nit=False):
    """Filter colleges based on criteria"""
    df = df.copy()
    
    df['close rank'] = pd.to_numeric(df['close rank'], errors='coerce')
    df = df.dropna(subset=['close rank'])
    
    # Show only colleges where Close Rank >= User's Rank (user can get admission)
    filters = (
        (df['gender'].str.lower().str.strip() == gender.lower().strip()) &
        (df['category'].str.lower().str.strip() == category.lower().strip()) &
        (df['close rank'] >= float(rank)) &
        (df['degree'].isin(degrees)) &
        (df['branch'].isin(branches))
    )
    
    df_filtered = df[filters].copy()
    
    if df_filtered.empty:
        return df_filtered[['college name', 'close rank']]
    
    if is_nit and state:
        def should_include_college(row):
            college_state = str(row.get('college state', '')).lower().strip()
            user_state = str(state).lower().strip()
            quota = str(row.get('quota', '')).upper().strip()
            
            if college_state == user_state:
                return quota == 'HS'
            else:
                return quota == 'OS'
        
        df_filtered = df_filtered[df_filtered.apply(should_include_college, axis=1)].copy()
        
        if df_filtered.empty:
            return df_filtered[['college name', 'close rank']]
        
        df_filtered = df_filtered.sort_values(by='close rank', ascending=True)
    else:
        df_filtered = df_filtered.sort_values(by='close rank', ascending=True)
    
    return df_filtered[['college name', 'close rank']].reset_index(drop=True)

def save_user_chat_json(user_data, nits_results, iiits_results):
    """Save complete user chat data to JSON file"""
    try:
        # In production, you might want to save to a database instead
        # For now, we'll just return a filename without actually saving
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in user_data['name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"user_chat_{safe_name}_{timestamp_str}.json"
        
        # In a real deployment, you'd save to cloud storage or database
        print(f"Would save chat data to: {filename}")
        
        return filename
        
    except Exception as e:
        print(f"Error saving chat: {e}")
        return None

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load sheets data on startup"""
    global sheets_data
    try:
        sheets_data = load_sheets_data()
        print("✅ Sheets data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load sheets data: {e}")
        # Continue startup even if sheets fail to load

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "College Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "get_options": "/options",
            "get_recommendations": "/recommendations",
            "health_check": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_loaded": len(sheets_data) > 0,
        "sheets_count": len(sheets_data),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/options", response_model=OptionsResponse)
async def get_options():
    """Get all dropdown options for the frontend"""
    return OptionsResponse(
        genders=GENDER_OPTIONS,
        categories=CATEGORY_OPTIONS,
        states=STATE_OPTIONS,
        degrees=DEGREE_OPTIONS,
        branches=BRANCH_OPTIONS
    )

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(preferences: UserPreferences):
    """Get college recommendations based on user preferences"""
    try:
        if not sheets_data:
            raise HTTPException(status_code=503, detail="Sheets data not loaded")
        
        # Filter NITs
        nits_df = filter_colleges(
            sheets_data.get("nits round 5", pd.DataFrame()),
            preferences.gender,
            preferences.category,
            preferences.rank,
            preferences.degrees,
            preferences.branches,
            state=preferences.state,
            is_nit=True
        )
        
        # Filter IIITs
        iiits_df = filter_colleges(
            sheets_data.get("iiits round 5", pd.DataFrame()),
            preferences.gender,
            preferences.category,
            preferences.rank,
            preferences.degrees,
            preferences.branches,
            state=None,
            is_nit=False
        )
        
        # Convert to list of dictionaries
        nits_results = []
        if not nits_df.empty:
            nits_results = [
                CollegeResult(
                    college_name=row['college name'],
                    close_rank=int(row['close rank'])
                ) for _, row in nits_df.iterrows()
            ]
        
        iiits_results = []
        if not iiits_df.empty:
            iiits_results = [
                CollegeResult(
                    college_name=row['college name'],
                    close_rank=int(row['close rank'])
                ) for _, row in iiits_df.iterrows()
            ]
        
        # Save user data
        user_data = {
            'name': preferences.name,
            'phone': preferences.phone,
            'gender': preferences.gender,
            'category': preferences.category,
            'state': preferences.state,
            'degrees': preferences.degrees,
            'branches': preferences.branches,
            'rank': preferences.rank
        }
        
        saved_file = None
        try:
            saved_file = save_user_chat_json(user_data, nits_df, iiits_df)
        except Exception as e:
            print(f"Warning: Could not save user data: {e}")
        
        return RecommendationResponse(
            success=True,
            message="Recommendations generated successfully",
            nits=nits_results,
            iiits=iiits_results,
            nit_count=len(nits_results),
            iiit_count=len(iiits_results),
            saved_file=saved_file
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/data/refresh")
async def refresh_data():
    """Refresh sheets data manually"""
    global sheets_data
    try:
        sheets_data = load_sheets_data()
        return {
            "success": True,
            "message": "Data refreshed successfully",
            "sheets_loaded": list(sheets_data.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)