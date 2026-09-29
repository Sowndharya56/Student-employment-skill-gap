"""Tests for improvement priority ranking and formula calculations."""
import pytest
from src.improvement_priority import calculate_improvement_priorities, calculate_percentile_rank


def test_calculate_percentile_rank():
    assert calculate_percentile_rank(2.0, 2.0, 4.0, 3.0) == 0.01
    assert calculate_percentile_rank(4.0, 2.0, 4.0, 3.0) == 0.99
    assert calculate_percentile_rank(3.0, 2.0, 4.0, 3.0) == 0.50


def test_calculate_improvement_priorities():
    weak_student = {
        'GPA': 2.1,
        'Credits_Completed': 30,
        'Exam_Scores': 52,
        'Attendance_Percentage': 55.0,
        'Coding_Skills': 2,
        'Communication_Skills': 2,
        'Leadership_Skills': 2,
        'Clubs_Joined': 0,
        'Events_Participated': 1,
        'Internship_Completed': 0,
        'Internship_Duration_Months': 0,
        'Library_Visits_Per_Week': 1,
        'Online_Course_Hours_Per_Week': 1
    }
    priorities = calculate_improvement_priorities(weak_student)
    assert len(priorities) == 13
    assert priorities[0]["rank"] == 1
    assert priorities[0]["priority_score"] >= priorities[-1]["priority_score"]
    for p in priorities:
        assert "feature_display" in p
        assert "dataset_position" in p
        assert "importance_level" in p
        assert "reason" in p
        assert 0.0 <= p["priority_score"] <= 100.0
