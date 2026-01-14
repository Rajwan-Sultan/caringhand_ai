
from fastapi import HTTPException, UploadFile
from ..schema.parser_schema import (ContactInfo,OtherLinksItem,EducationItem,ExperienceItem,DocumentSchema)
from ..core.config import settings
import asyncio
import json
import re
from google import genai
from google.genai import types




def get_mime_type(file: UploadFile) -> str:
    content_type = file.content_type.lower()
    
    mime_mapping = {
    "application/pdf": "application/pdf",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "text/plain": "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword": "application/msword"
}
    return mime_mapping.get(content_type, content_type)

# --- Text Extraction for Regex Fallback ---
async def extract_text_for_regex(file_content: bytes, mime_type: str) -> str:
    """Extract text from file for regex processing."""
    try:
        if mime_type == "application/pdf":
            # PDF text extraction with appropriate library
            if USING_PYMUPDF:
                # Use PyMuPDF for PDF text extraction
                doc = fitz.open(stream=file_content, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            else:
                # Use PyPDF2 as fallback
                from io import BytesIO
                reader = PyPDF2.PdfReader(BytesIO(file_content))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        elif mime_type in ["text/plain"]:
            return file_content.decode('utf-8', errors='ignore')
        #! Start - Eddited v1
        elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            # Extract text from DOCX/DOC
            if USING_DOCX:
                from io import BytesIO
                doc = docx.Document(BytesIO(file_content))
                text = "\n".join([para.text for para in doc.paragraphs])
                return text
            else:
                # Basic fallback for Word docs when docx module is not available
                return "Word document - text extraction requires python-docx module"
        #! End - Eddited v1
        elif mime_type.startswith("image/"):
            # For images, return empty string as regex can't extract text from images
            return "Image file - text extraction not available for regex fallback"
        else:
            # Try to decode as text for other formats
            return file_content.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Text extraction error: {e}")
        return "Text extraction failed"

def extract_with_regex(text: str) -> DocumentSchema:
    name_pattern = r'^([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s*$'
    name = "Unknown"
    for line in text.split('\n')[:10]:  # Check first 10 lines
        line = line.strip()
        if line and re.match(name_pattern, line):
            name = line
            break
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text, re.IGNORECASE)
    
    # Extract phone numbers (improved pattern)
    phone_patterns = [
        r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
        r'(?:\+?91[-.\s]?)?([0-9]{10})',
        r'(?:\+?[1-9][0-9]{0,3}[-.\s]?)?([0-9]{4,})[-.\s]?([0-9]{4,})'
    ]
    
    phone_numbers = []
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    phone_numbers.append('-'.join(match))
                else:
                    phone_numbers.append(match)
    
    if not phone_numbers:
        phone_numbers = ["Not found"]
    
    # Extract skills (expanded list)
    skills_keywords = [
        'python', 'java', 'javascript', 'react', 'angular', 'node', 'sql', 'html', 'css', 'git', 
        'docker', 'aws', 'azure', 'kubernetes', 'tensorflow', 'pytorch', 'django', 'flask',
        'spring', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'jenkins',
        'linux', 'windows', 'macos', 'android', 'ios', 'flutter', 'react native', 'vue',
        'typescript', 'php', 'ruby', 'go', 'rust', 'scala', 'kotlin', 'swift','developer',
        'construction','management','nursing','accounting','marketing','sales','communication',
        'leadership','teamwork','problem-solving','time management','adaptability','creativity',
        'critical thinking','customer service','data analysis','project management','finance'
    ]
    
    found_skills = []
    text_lower = text.lower()
    for skill in skills_keywords:
        if skill in text_lower:
            found_skills.append(skill.title())
    
    if not found_skills:
        found_skills = ["Not specified"]
    
    # Create contact info
    contact_info = []
    if emails:
        try:
            contact_info.append(ContactInfo(
                email=emails[0],
                phone=phone_numbers[0] if phone_numbers else "Not found",
                address="Not specified"
            ))
        except Exception:
            contact_info.append(ContactInfo(
                email="unknown@example.com",
                phone=phone_numbers[0] if phone_numbers else "Not found",
                address="Not specified"
            ))
    else:
        contact_info.append(ContactInfo(
            email="unknown@example.com",
            phone=phone_numbers[0] if phone_numbers else "Not found",
            address="Not specified"
        ))
    
    return DocumentSchema(
        llm_parsing_successful=0,  # Regex fallback
        name=name,
        contact_info=contact_info,
        other_links=None,
        education=None,
        experience=None,
        skills=found_skills,
        languages=None
    )


async def parse_with_llm(file_content: bytes, mime_type: str) -> DocumentSchema:
    try:
        gemini_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    except Exception as e:
        print(f"Warning: Gemini client initialization failed: {e}")

    if not gemini_client:
        raise Exception("Gemini client not available")
    
    format_instructions = (
        "Return valid JSON with these exact fields:\n"
        "Strictly use this format:\n"
        f"{DocumentSchema.model_json_schema()}"
    )
    
    prompt = (
        "Extract resume data and return ONLY valid JSON. No explanations or formatting.\n"
        "RULE - Do not make any information on your own. If not found then leave it empty and try to extract infromation as accurately as possible\n"
        "Focus on relevant sections like personal info, contact details, education, experience, skills, languages,starting date,ending date,education..\n"
        "If any information is missing, then just leave it empty or use empty arrays.\n"
        "Ignore irrelevant sections like references, hobbies, or template artifacts.\n\n"
        f"{format_instructions}\n\n"
    )
    
    #! Start - Eddited v1
    # Create parts for multimodal input
    supported_mimes = ["application/pdf", "image/jpeg", "image/png", "text/plain"]
    if mime_type not in supported_mimes:
        text_content = await extract_text_for_regex(file_content, mime_type)
        parts = [
            types.Part.from_text(text=text_content),
            types.Part.from_text(text=prompt)
        ]
    else:
        parts = [
            types.Part.from_bytes(
                data=file_content, 
                mime_type=mime_type
            ),
            types.Part.from_text(text=prompt)
        ]
    # parts = [
    #     types.Part.from_bytes(
    #         data=file_content,
    #         mime_type=mime_type
    #     ),
    #     types.Part.from_text(text=prompt)
    # ]
    #! End - Eddited v1
    
    # Create generation config with temperature and max tokens
    generation_config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=2000, 
        response_mime_type="application/json",
        response_schema=DocumentSchema.model_json_schema()
        # top_p=0.8,  # Optional: nucleus sampling
        # top_k=40,   # Optional: top-k sampling
    )
    
    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=parts,
        config=generation_config
    )





    if not response.text:
        raise Exception("Empty LLM response")
    
    # Clean and parse JSON response
    json_text = response.text.strip()
    if json_text.startswith('```json'):
        json_text = json_text[7:]
    if json_text.endswith('```'):
        json_text = json_text[:-3]
    
    parsed_json = json.loads(json_text)
    return DocumentSchema.model_validate(parsed_json)
