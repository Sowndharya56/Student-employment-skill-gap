"""Recommended learning areas module based strictly on measurable student factors."""
from typing import Dict, List, Any, Optional

# Structured mapping of directly measurable student factors to foundational learning areas
FACTOR_LEARNING_AREAS = {
    "Coding_Skills": {
        "title": "Computer Programming & Algorithmic Problem Solving",
        "focus_areas": [
            "Programming fundamentals and syntax mastery",
            "Data structures (Arrays, Linked Lists, Trees, Hash Maps)",
            "Algorithm design, recursion, and time/space complexity",
            "Practical coding practice on competitive platforms and GitHub projects"
        ],
        "action_item": "Dedicate consistent weekly time to solve programming challenges and write clean, modular code."
    },
    "Communication_Skills": {
        "title": "Professional & Technical Communication",
        "focus_areas": [
            "Technical documentation, architecture write-ups, and pull request summaries",
            "Presentation design and technical speaking",
            "Structured interview communication (STAR methodology)",
            "Active listening and cross-functional team collaboration"
        ],
        "action_item": "Participate in mock technical interviews and present project demos in student seminars."
    },
    "Leadership_Skills": {
        "title": "Team Leadership & Project Ownership",
        "focus_areas": [
            "Agile/Scrum team coordination and sprint planning",
            "Conflict resolution and mentorship of junior peers",
            "End-to-end milestone ownership for group assignments and hackathons",
            "Delegation, code review etiquette, and peer feedback"
        ],
        "action_item": "Take initiative as a lead contributor or team coordinator in collaborative academic and club projects."
    },
    "Internship_Completed": {
        "title": "Practical Industry Exposure & Internship Preparation",
        "focus_areas": [
            "Portfolio building with deployed full-stack or data applications",
            "Resume tailoring and cold outreach for trainee/internship positions",
            "Familiarity with industry version control workflows (Git branching, PR reviews)",
            "Understanding workplace development standards and unit testing"
        ],
        "action_item": "Apply to internship postings and contribute to active open-source repositories to build verifiable experience."
    },
    "Internship_Duration_Months": {
        "title": "Extended Industry Experience & Practical Exposure",
        "focus_areas": [
            "Long-term engineering project contributions",
            "Experience with production software lifecycles and deployments",
            "Cross-departmental collaboration with product and QA teams",
            "Building measurable business impact case studies for resumes"
        ],
        "action_item": "Seek multi-month research or industry internships to deepen practical engineering tenure."
    },
    "Online_Course_Hours_Per_Week": {
        "title": "Structured Self-Paced Continuous Upskilling",
        "focus_areas": [
            "Establishing a disciplined weekly learning schedule",
            "Taking recognized university or industry-standard specialization curricula",
            "Hands-on lab exercises accompanying theoretical study",
            "Summarizing learning into personal tech blogs or repository notes"
        ],
        "action_item": "Allocate 5 to 10 structured hours per week to systematically complete technical specializations."
    },
    "GPA": {
        "title": "Academic Rigor & Core Curriculum Foundations",
        "focus_areas": [
            "Core computer science and engineering coursework (OS, DBMS, Networks)",
            "Consistent exam preparation and conceptual review",
            "Office hour consultations with faculty for challenging subject areas",
            "Peer study groups to reinforce difficult theoretical topics"
        ],
        "action_item": "Focus on improving grades in foundational technical courses to strengthen overall academic profile."
    },
    "Exam_Scores": {
        "title": "Subject Mastery & Technical Examination Readiness",
        "focus_areas": [
            "Reviewing past exam papers and standard technical assessments",
            "Clarifying foundational concepts in math, logic, and algorithms",
            "Time management strategies during technical written tests",
            "Formulating concise answers for theoretical questions"
        ],
        "action_item": "Target weak subject areas identified from previous semester exam score breakdowns."
    },
    "Attendance_Percentage": {
        "title": "Classroom Engagement & Academic Discipline",
        "focus_areas": [
            "Regular lecture and laboratory attendance",
            "Active in-class participation and questioning",
            "Direct engagement with faculty discussions and live coding sessions"
        ],
        "action_item": "Maintain consistent daily attendance to avoid falling behind on complex syllabus topics."
    },
    "Clubs_Joined": {
        "title": "Extracurricular & Technical Community Engagement",
        "focus_areas": [
            "Joining coding clubs, robotics societies, or student developer circles",
            "Organizing campus workshops and peer-learning events",
            "Networking with alumni and senior students in student chapters"
        ],
        "action_item": "Actively participate in at least one university technical club or developer chapter."
    },
    "Events_Participated": {
        "title": "Hackathons, Competitions & Technical Events",
        "focus_areas": [
            "Hackathon participation for rapid prototyping under time constraints",
            "University technical symposiums and coding competitions",
            "Presenting project papers or posters at regional conferences"
        ],
        "action_item": "Register for upcoming inter-college hackathons to test skills in competitive environments."
    },
    "Library_Visits_Per_Week": {
        "title": "Focused Study & Research Exploration",
        "focus_areas": [
            "Dedicated distraction-free study blocks for deep work",
            "Consulting reference textbooks and IEEE/ACM research publications",
            "Utilizing quiet study spaces for project documentation and thesis writing"
        ],
        "action_item": "Incorporate scheduled quiet study blocks into the weekly academic routine."
    },
    "Credits_Completed": {
        "title": "Curricular Progression & Degree Completion",
        "focus_areas": [
            "Tracking degree requirements and elective prerequisites",
            "Balancing semester credit loads to maintain academic performance",
            "Timely enrollment in required capstone project courses"
        ],
        "action_item": "Consult academic advisor to ensure steady progress toward credit completion milestones."
    }
}


def get_learning_recommendations_for_priorities(
    priorities: List[Dict[str, Any]],
    top_n: int = 4
) -> List[Dict[str, Any]]:
    """Retrieve structured learning recommendations for the student's top improvement priorities.
    
    Args:
        priorities: Ranked list of improvement priorities from calculate_improvement_priorities().
        top_n: Number of top priority areas to generate recommendations for.
        
    Returns:
        List of recommendation dictionaries.
    """
    recommendations = []
    
    for item in priorities[:top_n]:
        feature_name = item["feature"]
        if feature_name in FACTOR_LEARNING_AREAS:
            data = FACTOR_LEARNING_AREAS[feature_name]
            recommendations.append({
                "rank": item.get("rank", len(recommendations) + 1),
                "feature": feature_name,
                "feature_display": item.get("feature_display", feature_name.replace('_', ' ')),
                "learning_area_title": data["title"],
                "focus_areas": data["focus_areas"],
                "action_item": data["action_item"],
                "priority_score": item.get("priority_score", 0),
                "importance_level": item.get("importance_level", "Normal"),
                "reason": item.get("reason", "")
            })
            
    return recommendations
