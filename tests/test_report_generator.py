"""Tests for learning recommendations and PDF report generation."""
import pytest
from src.improvement_priority import calculate_improvement_priorities
from src.learning_recommendations import get_learning_recommendations_for_priorities
from src.prediction import predict_single_student
from src.feature_importance import get_top_contributing_factors
from src.industry_skills import get_industry_skill_analyzer
from src.report_generator import generate_pdf_report


def test_learning_recommendations():
    sample_student = {
        'GPA': 2.5,
        'Credits_Completed': 40,
        'Exam_Scores': 60,
        'Attendance_Percentage': 65.0,
        'Coding_Skills': 3,
        'Communication_Skills': 4,
        'Leadership_Skills': 3,
        'Clubs_Joined': 1,
        'Events_Participated': 2,
        'Internship_Completed': 0,
        'Internship_Duration_Months': 0,
        'Library_Visits_Per_Week': 2,
        'Online_Course_Hours_Per_Week': 3
    }
    priorities = calculate_improvement_priorities(sample_student)
    recs = get_learning_recommendations_for_priorities(priorities, top_n=3)
    assert len(recs) == 3
    for r in recs:
        assert "learning_area_title" in r
        assert len(r["focus_areas"]) > 0
        assert "action_item" in r


def test_generate_pdf_report_with_student_name_and_department():
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
    pred = predict_single_student(sample_student)
    top_factors = get_top_contributing_factors(5)
    analyzer = get_industry_skill_analyzer()
    role_info = analyzer.get_role_details(".NET Developer")
    priorities = calculate_improvement_priorities(sample_student)
    recs = get_learning_recommendations_for_priorities(priorities, top_n=3)

    # Test calling pattern 1: keyword arguments with student_name and department at front
    pdf_bytes_1 = generate_pdf_report(
        student_name="Yuvan Tech",
        department="Cyber Security",
        student_profile=sample_student,
        prediction_result=pred,
        top_factors=top_factors,
        target_role=role_info,
        priorities=priorities,
        recommendations=recs
    )
    
    assert isinstance(pdf_bytes_1, bytes)
    assert len(pdf_bytes_1) > 2000
    assert pdf_bytes_1.startswith(b'%PDF')

    # Test calling pattern 2: keyword arguments with student_profile first
    pdf_bytes_2 = generate_pdf_report(
        student_profile=sample_student,
        prediction_result=pred,
        top_factors=top_factors,
        target_role=role_info,
        priorities=priorities,
        recommendations=recs,
        student_name="Yuvan Tech",
        department="Cyber Security"
    )
    
    assert isinstance(pdf_bytes_2, bytes)
    assert len(pdf_bytes_2) > 2000
    assert pdf_bytes_2.startswith(b'%PDF')
