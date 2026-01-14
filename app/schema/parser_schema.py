from pydantic import BaseModel, Field
from typing import Optional, List, Any, Annotated



# --- Data Models (Unchanged as requested) ---
class ContactInfo(BaseModel):
    email: Annotated[Any, Field(description="Email address of the individual")]
    phone: Annotated[Any, Field(description="Phone number of the individual")]
    address: Annotated[Any, Field(description="Physical address of the individual")]

class OtherLinksItem(BaseModel):
    link_type: Annotated[Any, Field(description="Type of the link, e.g., LinkedIn, GitHub, Portfolio")]
    url: Annotated[Any, Field(description="URL of the link")]

class EducationItem(BaseModel):
    institution: Annotated[Any, Field(description="Name of the educational institution")]
    degree: Annotated[Any, Field(description="Degree obtained, e.g., B.Sc, M.Sc, PhD")]
    field_of_study: Annotated[Any, Field(description="Field of study, e.g., Computer Science, Business")]
    results: Annotated[Any, Field(description="Results/Grade/Percentage/CGPA")]
    start_date: Annotated[Optional[str], Field(default=None, description="Start date of the education in YYYY-MM format")]
    end_date: Annotated[Optional[str], Field(default=None, description="End date of the education in YYYY-MM format or 'Present' if ongoing")]

class ExperienceItem(BaseModel):
    company: Annotated[Any, Field(description="Name of the company")]
    designation: Annotated[Any, Field(description="Designation/role of the individual at the company")]
    details: Annotated[Optional[str], Field(default=None, description="Detailed description of roles and responsibilities")]
    start_date: Annotated[Optional[str], Field(default=None, description="Start date of the experience in YYYY-MM format")]
    end_date: Annotated[Optional[str], Field(default=None, description="End date of the experience in YYYY-MM format or 'Present' if ongoing")]

class DocumentSchema(BaseModel):
    llm_parsing_successful: Annotated[int, Field(description="Indicator if LLM parsing was successful (1) or not (0)")]
    name: Annotated[str, Field(description="Full name of the individual")]
    #profile_description: Annotated[Optional[str], Field(default=None, description="Brief profile description or summary of the individual given in the resume")]
    profile_detailed_information: Annotated[Optional[str], Field(default=None, description="This field contains the information of the overview/profile summary section of the resume in detail.If not found then leave it empty")]
    designation: Annotated[Optional[str], Field(default=None, description="Professional designation or title of the individual")]
    contact_info: Annotated[List[ContactInfo], Field(description="Contact information including email, phone, and address")]
    other_links: Annotated[Optional[List[OtherLinksItem]], Field(default=None, description="Other relevant links such as LinkedIn, GitHub, Portfolio")]
    education: Annotated[Optional[List[EducationItem]], Field(default=None, description="Educational background including institution,degree,field of study,results")]
    experience: Annotated[Optional[List[ExperienceItem]], Field(default=None, description="List of work experiences including only company name and designation")]
    skills: Annotated[Optional[List[str]], Field(default = None,description="List of skills containing skill name")]
    languages: Annotated[Optional[List[str]], Field(default=None, description="The name of the languages known")]



