from fastapi import APIRouter, HTTPException,UploadFile, File,status
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from ..schema.parser_schema import (ContactInfo,OtherLinksItem,EducationItem,ExperienceItem,DocumentSchema)
from ..utils.resume_parser import parse_with_llm,get_mime_type,extract_text_for_regex,extract_with_regex
router = APIRouter()

@router.post("/parse-resume", response_model=DocumentSchema)
async def parse_resume(file: UploadFile = File(...)):
    """Parse resume with LLM fallback to regex."""
    MAX_FILE_SIZE =5 * 1024 * 1024 # 5MB
    # Validate file size
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size allowed: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Validate file type
    allowed_types = [
        "text/plain", 
        "application/pdf", 
        "image/jpeg", 
        "image/png", 
        # "image/gif",
        # "image/webp",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported types: {allowed_types}"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Double-check file size after reading (in case file.size was None)
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size allowed: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Get proper MIME type
        mime_type = get_mime_type(file)
        
        # Try LLM first with native file processing
        try:
            result = await parse_with_llm(file_content, mime_type)
            return result
        except Exception as llm_error:
            print(f"LLM parsing failed: {llm_error}")
            
            # Fallback to regex - extract text first
            try:
                text_content = await extract_text_for_regex(file_content, mime_type)
                result = extract_with_regex(text_content)
                return result
            except Exception as regex_error:
                print(f"Regex fallback also failed: {regex_error}")
                
                # Final fallback - return minimal structure
                return DocumentSchema(
                    llm_parsing_successful=0,
                    name="Unknown",
                    contact_info=[ContactInfo(
                        email="unknown@example.com",
                        phone="Not found",
                        address="Not specified"
                    )],
                    other_links=None,
                    education=None,
                    experience=None,
                    skills=["Not specified"],
                    languages=None
                )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await file.close()
