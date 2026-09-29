"""Tests for industry skill analysis, role retrieval, and skill alignment."""
import pytest
from src.industry_skills import get_industry_skill_analyzer, IndustrySkillAnalyzer


def test_industry_skills_loading():
    analyzer = get_industry_skill_analyzer()
    roles = analyzer.get_all_job_roles()
    assert len(roles) >= 460
    assert "Cloud Architect" in roles or ".NET Developer" in roles


def test_get_role_details():
    analyzer = get_industry_skill_analyzer()
    role_info = analyzer.get_role_details(".NET Developer")
    assert role_info is not None
    assert role_info["job_title"] == ".NET Developer"
    assert len(role_info["skills"]) > 0
    assert any("C#" in s or ".NET" in s for s in role_info["skills"])


def test_align_student_with_role():
    analyzer = get_industry_skill_analyzer()
    sample_student = {
        'GPA': 3.4,
        'Credits_Completed': 90,
        'Exam_Scores': 78,
        'Attendance_Percentage': 82.0,
        'Coding_Skills': 7,
        'Communication_Skills': 6,
        'Leadership_Skills': 5,
        'Clubs_Joined': 2,
        'Events_Participated': 4,
        'Internship_Completed': 1,
        'Internship_Duration_Months': 6,
        'Library_Visits_Per_Week': 6,
        'Online_Course_Hours_Per_Week': 10
    }
    alignment = analyzer.align_student_with_role(".NET Developer", sample_student)
    assert alignment["job_title"] == ".NET Developer"
    assert len(alignment["required_industry_skills"]) > 0
    assert len(alignment["unmeasured_skills"]) > 0
    for u in alignment["unmeasured_skills"]:
        assert u["measurement_status"] == "Not directly measured by the current student dataset"
    assert len(alignment["measured_student_factors"]) > 0
