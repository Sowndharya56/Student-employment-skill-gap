"""Industry job roles, skill extraction, and skill alignment module."""
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import pandas as pd

from src.data_loader import load_job_dataset
from src.data_cleaning import clean_job_data


class IndustrySkillAnalyzer:
    """Analyzer for industry IT job roles, extracted skills, and certifications."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.job_df = self._load_and_prepare_data()
        self.roles_cache = self._build_roles_cache()
        
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load and clean IT job roles dataset."""
        raw_df = load_job_dataset(self.dataset_path)
        return clean_job_data(raw_df)
        
    def _build_roles_cache(self) -> Dict[str, Dict[str, Any]]:
        """Index job roles by title, aggregating multiple records where applicable."""
        cache = {}
        for _, row in self.job_df.iterrows():
            title = str(row['Job Title']).strip()
            desc = str(row['Job Description']).strip()
            skills_raw = str(row['Skills'])
            certs_raw = str(row['Certifications'])
            
            parsed_skills = self.parse_items(skills_raw)
            parsed_certs = self.parse_items(certs_raw)
            
            if title not in cache:
                cache[title] = {
                    "job_title": title,
                    "descriptions": [desc] if desc else [],
                    "skills": list(parsed_skills),
                    "certifications": list(parsed_certs)
                }
            else:
                # Merge unique descriptions, skills, and certifications
                if desc and desc not in cache[title]["descriptions"]:
                    cache[title]["descriptions"].append(desc)
                for s in parsed_skills:
                    if s not in cache[title]["skills"]:
                        cache[title]["skills"].append(s)
                for c in parsed_certs:
                    if c not in cache[title]["certifications"]:
                        cache[title]["certifications"].append(c)
                        
        return cache
        
    @staticmethod
    def normalize_skill(skill: str) -> str:
        """Normalize skill string for case-insensitive comparison."""
        return skill.strip().lower()
        
    @staticmethod
    def parse_items(raw_text: str) -> List[str]:
        """Parse comma-separated or semicolon-separated items, cleaning artifacts."""
        if not raw_text or pd.isna(raw_text) or str(raw_text).strip() == '':
            return []
            
        cleaned_text = str(raw_text).replace(';', ',')
        # Clean potential mojibake dashes
        cleaned_text = cleaned_text.replace('\x96', '-').replace('\x97', '-').replace('\x80\x93', '-')
        
        parts = [p.strip() for p in cleaned_text.split(',') if p.strip()]
        # Deduplicate while preserving case and order
        seen = set()
        deduped = []
        for p in parts:
            norm = p.lower()
            if norm not in seen:
                seen.add(norm)
                deduped.append(p)
        return deduped

    def get_all_job_roles(self) -> List[str]:
        """Retrieve sorted list of all unique job titles from the dataset."""
        return sorted(list(self.roles_cache.keys()))

    def get_role_details(self, job_title: str) -> Optional[Dict[str, Any]]:
        """Retrieve aggregated role details for a given job title.
        
        Args:
            job_title: Job title string.
            
        Returns:
            Dict containing job title, consolidated description, skills, and certifications.
        """
        if job_title in self.roles_cache:
            data = self.roles_cache[job_title]
            return {
                "job_title": data["job_title"],
                "job_description": " ".join(data["descriptions"]),
                "skills": data["skills"],
                "certifications": data["certifications"]
            }
            
        # Case-insensitive lookup fallback
        norm_query = job_title.strip().lower()
        for title, data in self.roles_cache.items():
            if title.strip().lower() == norm_query:
                return {
                    "job_title": data["job_title"],
                    "job_description": " ".join(data["descriptions"]),
                    "skills": data["skills"],
                    "certifications": data["certifications"]
                }
        return None

    def get_skill_frequency(self) -> Dict[str, int]:
        """Compute frequency distribution of all skills across the job roles dataset."""
        counter = Counter()
        for role_data in self.roles_cache.values():
            for skill in role_data["skills"]:
                counter[skill] += 1
        return dict(counter.most_common())

    def align_student_with_role(
        self,
        job_title: str,
        student_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform transparent skill alignment between student measurable factors and role requirements.
        
        Args:
            job_title: Target job title.
            student_profile: Student input dictionary.
            
        Returns:
            Dict containing role requirements, directly measured factors, and unmeasured industry skills.
        """
        role_details = self.get_role_details(job_title)
        if not role_details:
            return {
                "error": f"Job title '{job_title}' not found in dataset.",
                "available_in_dataset": False
            }
            
        industry_skills = role_details["skills"]
        
        # Categorize industry technical skills as unmeasured directly in student dataset
        unmeasured_industry_skills = []
        for s in industry_skills:
            unmeasured_industry_skills.append({
                "skill_name": s,
                "measurement_status": "Not directly measured by the current student dataset",
                "relevance": "Required by selected industry job role"
            })
            
        # Transparently map measurable student factors to job readiness dimensions
        measured_factors = [
            {
                "factor_name": "Coding Skills",
                "current_value": student_profile.get("Coding_Skills", "N/A"),
                "scale": "1 - 9",
                "concept_alignment": "General programming readiness and analytical problem solving",
                "measurement_status": "Directly Measured (General Readiness)"
            },
            {
                "factor_name": "Communication Skills",
                "current_value": student_profile.get("Communication_Skills", "N/A"),
                "scale": "1 - 9",
                "concept_alignment": "Interpersonal, team collaboration, and client communication",
                "measurement_status": "Directly Measured (General Readiness)"
            },
            {
                "factor_name": "Leadership Skills",
                "current_value": student_profile.get("Leadership_Skills", "N/A"),
                "scale": "1 - 9",
                "concept_alignment": "Project ownership, leadership, and club/team initiative",
                "measurement_status": "Directly Measured (General Readiness)"
            },
            {
                "factor_name": "Internship Experience",
                "current_value": f"{student_profile.get('Internship_Duration_Months', 0)} months (Completed: {'Yes' if student_profile.get('Internship_Completed', 0) == 1 else 'No'})",
                "scale": "0 - 11 months",
                "concept_alignment": "Practical industry exposure and workplace project experience",
                "measurement_status": "Directly Measured (General Readiness)"
            },
            {
                "factor_name": "Online Course Hours",
                "current_value": f"{student_profile.get('Online_Course_Hours_Per_Week', 'N/A')} hrs/week",
                "scale": "0 - 19 hrs/week",
                "concept_alignment": "Continuous self-learning and upskilling commitment",
                "measurement_status": "Directly Measured (General Readiness)"
            },
            {
                "factor_name": "Academic Foundations (GPA & Exam Scores)",
                "current_value": f"GPA: {student_profile.get('GPA', 'N/A')}, Exam: {student_profile.get('Exam_Scores', 'N/A')}",
                "scale": "GPA: 2.0-4.0, Exam: 50-99",
                "concept_alignment": "Core academic rigor and theoretical foundations",
                "measurement_status": "Directly Measured (General Readiness)"
            }
        ]
        
        return {
            "job_title": role_details["job_title"],
            "job_description": role_details["job_description"],
            "required_industry_skills": industry_skills,
            "required_certifications": role_details["certifications"],
            "unmeasured_skills": unmeasured_industry_skills,
            "measured_student_factors": measured_factors,
            "alignment_note": "Direct individual technical skills (e.g. Python, AWS, Docker) are not individually scored in the student dataset and are classified as unmeasured."
        }


# Global singleton instance helper
_analyzer_instance = None


def get_industry_skill_analyzer() -> IndustrySkillAnalyzer:
    """Get or instantiate singleton IndustrySkillAnalyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = IndustrySkillAnalyzer()
    return _analyzer_instance
