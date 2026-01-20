from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String, index=True)  # UUID stored as text in database
    # Add other fields as needed


class DocumentProcessedData(Base):
    __tablename__ = "document_processed_data"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, index=True)  # UUID stored as text
    project_id = Column(String, index=True)  # UUID stored as text
    result = Column(String)  # JSON stored as text


class ProjectResult(Base):
    __tablename__ = "project_results"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String, unique=True, index=True)  # UUID stored as text
    charts_data = Column(String)  # JSON stored as text
    created_at = Column(String)  # Timestamp as text
    updated_at = Column(String)  # Timestamp as text


# Pydantic Models for API
class ProcessedDataResponse(BaseModel):
    document_id: str
    result: Optional[dict]
    
    class Config:
        from_attributes = True


class ChartData(BaseModel):
    chart_type: str  # "histogram" or "bar_chart"
    title: str
    labels: List[str]
    values: List[float]


class OrganizedResultsResponse(BaseModel):
    project_id: UUID
    charts: List[ChartData]


class ProjectResultsResponse(BaseModel):
    project_id: UUID
    results: List[ProcessedDataResponse]


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# FastAPI app
app = FastAPI(title="Document Processing API")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables on startup
@app.on_event("startup")
def create_tables():
    """Create all tables if they don't exist"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")


@app.get("/")
def read_root():
    return {"message": "Document Processing API"}


def organize_results_with_openai(results: List[dict]) -> List[ChartData]:
    """
    Use OpenAI to analyze results and organize them for visualization.
    """
    prompt = f"""You are a data visualization expert. Extract and organize ONLY NUMERIC DATA from the JSON below into meaningful, well-structured charts.

CRITICAL RULES:
❌ IGNORE all text-only fields: URLs, emails, names, addresses, descriptions, phone numbers
✓ ONLY extract label-number pairs where value is a NUMBER for charts
✓ GROUP related metrics into logical chart categories
✓ SORT values in meaningful order (highest to lowest, or logical sequence)
✓ Use clear, concise chart titles
✓ Combine similar metrics into single charts when appropriate

DATA TO ANALYZE:
{json.dumps(results, indent=2)}

Organize into charts with this structure:
{{
    "charts": [
        {{
            "chart_type": "bar_chart",
            "title": "Student Enrollment by Institution Type",
            "labels": ["Junior Secondary", "Tertiary Public", "Private Institutions"],
            "values": [90000, 44000, 50421]
        }},
        {{
            "chart_type": "bar_chart",
            "title": "Education System Structure (Years)",
            "labels": ["Primary", "Junior Secondary", "Senior Secondary", "University"],
            "values": [6, 3, 3, 4]
        }},
        {{
            "chart_type": "bar_chart",
            "title": "Academic Grade Point Values",
            "labels": ["A", "B", "C", "D", "E", "S", "F"],
            "values": [6, 5, 4, 3, 2, 1, 0]
        }}
    ]
}}

BEST PRACTICES:
✓ Group similar metrics together
✓ Sort values logically (descending for comparisons, sequential for grades/time)
✓ Use descriptive but concise titles
✓ Ensure all values are actual numbers
✓ Create separate charts for different data domains
✓ If no numeric data exists, return {{"charts": []}}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract and organize ONLY numeric data for visualization. Group related metrics logically. Sort values meaningfully. Ignore all text-only fields. Return well-structured JSON with 'charts' array."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        
        # Extract charts array
        charts_data = parsed.get("charts", [])
        if not isinstance(charts_data, list):
            charts_data = []
            
        return [ChartData(**chart) for chart in charts_data]
    except Exception as e:
        print(f"Error organizing with OpenAI: {e}")
        return []


@app.get("/projects/{project_id}/results", response_model=OrganizedResultsResponse)
def get_project_results(project_id: UUID, db: Session = Depends(get_db)):
    """
    Get saved organized results for a specific project.
    
    Args:
        project_id: The UUID of the project
        db: Database session
        
    Returns:
        Saved organized chart data for the project
    """
    # Check if we have saved results for this project
    saved_result = db.query(ProjectResult).filter(
        ProjectResult.project_id == str(project_id)
    ).first()
    
    if not saved_result:
        raise HTTPException(
            status_code=404, 
            detail=f"No organized results found for project {project_id}. Use /organized-results endpoint first to generate results."
        )
    
    try:
        # Parse the saved charts data
        charts_data = json.loads(saved_result.charts_data)
        charts = [ChartData(**chart) for chart in charts_data]
        
        return OrganizedResultsResponse(
            project_id=project_id,
            charts=charts
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing saved results: {str(e)}"
        )


@app.get("/projects/{project_id}/organized-results", response_model=OrganizedResultsResponse)
def get_organized_project_results(project_id: UUID, db: Session = Depends(get_db)):
    """
    Generate and save organized results for visualization using OpenAI.
    
    Args:
        project_id: The UUID of the project
        db: Database session
        
    Returns:
        Organized chart data optimized for visualization (also saves to database)
    """
    # Query document_processed_data directly with project_id
    processed_data = db.query(DocumentProcessedData).filter(
        DocumentProcessedData.project_id == str(project_id)
    ).all()
    
    if not processed_data:
        raise HTTPException(
            status_code=404, 
            detail=f"No processed data found for project {project_id}"
        )
    
    # Prepare data for OpenAI analysis
    raw_data_for_openai = []
    
    for data in processed_data:
        # Parse result if it's a JSON string
        result_data = data.result
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
            except json.JSONDecodeError:
                result_data = None
        
        if result_data:
            raw_data_for_openai.append({
                "document_id": str(data.document_id),
                "data": result_data
            })
    
    # Use OpenAI to organize data for charts
    charts = organize_results_with_openai(raw_data_for_openai)
    
    # Save results to database
    try:
        charts_json = json.dumps([chart.dict() for chart in charts])
        current_time = datetime.now().isoformat()
        
        # Check if project result already exists
        existing_result = db.query(ProjectResult).filter(
            ProjectResult.project_id == str(project_id)
        ).first()
        
        if existing_result:
            # Update existing record
            existing_result.charts_data = charts_json
            existing_result.updated_at = current_time
        else:
            # Create new record
            new_result = ProjectResult(
                project_id=str(project_id),
                charts_data=charts_json,
                created_at=current_time,
                updated_at=current_time
            )
            db.add(new_result)
        
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error saving results to database: {e}")
        # Continue and return results even if save fails
    
    return OrganizedResultsResponse(
        project_id=project_id,
        charts=charts
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
