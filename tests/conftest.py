"""Shared fixtures: sample documents and helpers used by all tests."""

from __future__ import annotations

import io

import pymupdf
import pytest
from docx import Document


@pytest.fixture()
def fake_pdf_text_bytes() -> bytes:
    """A real single-page PDF generated in-memory containing resume text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Jane Doe\nEmail: jane.doe@example.com\nPhone: +1 555 010 1234")
    page.insert_text((72, 110), "SUMMARY")
    page.insert_text((72, 122), "Backend engineer focused on Python services.")
    page.insert_text((72, 150), "SKILLS")
    page.insert_text((72, 162), "Python, JavaScript (React), Kubernetes, AWS")
    page.insert_text((72, 190), "EXPERIENCE")
    page.insert_text((72, 202), "Built APIs that cut latency by 40% using Python and Docker.")
    page.insert_text((72, 230), "EDUCATION")
    page.insert_text((72, 242), "B.S. Computer Science, 2019")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def image_only_pdf_bytes() -> bytes:
    """A PDF whose page contains vector drawing but no selectable text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(40, 40, 300, 120), fill=(0, 0, 1))
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def fake_docx_bytes() -> bytes:
    """A real DOCX built in-memory with python-docx."""
    document = Document()
    document.add_heading("Jane Doe", 0)
    document.add_paragraph("jane.doe@example.com | +1 555 010 1234")
    document.add_heading("Summary", 1)
    document.add_paragraph("Backend engineer focused on Python services.")
    document.add_heading("Skills", 1)
    document.add_paragraph("Python, PostgreSQL, Docker, Kubernetes")
    document.add_heading("Experience", 1)
    document.add_paragraph("Built Python services; reduced latency by 40%.")
    target = io.BytesIO()
    document.save(target)
    return target.getvalue()


@pytest.fixture()
def sample_resume_text() -> str:
    return SAMPLE_RESUME_TEXT


@pytest.fixture()
def sample_jd_text() -> str:
    return SAMPLE_JD_TEXT


SAMPLE_RESUME_TEXT = """Jane Doe
jane.doe@example.com | +1 555 010 1234 | linkedin.com/in/janedoe

SUMMARY
Backend engineer with 6 years of experience building Python services and
data pipelines. Reduced deploy cycle by 40% and cut API costs by $120k/year.

EXPERIENCE
Senior Software Engineer, Dataworks — 2021-2026
- Built 12 Python fastapi read replicas; cut p99 latency 45% via query tuning.
- Migrated legacy ETL jobs to Spark and dbt, saving $120k per year.
- Mentored 4 engineers; led Kubernetes and Docker adoption across 3 teams.

Software Engineer, Bitsmith — 2019-2021
- Implementing REST APIs for invoicing with Python and PostgreSQL.
- Automated CI/CD pipelines with GitHub Actions; deploys went weekly to daily.

EDUCATION
B.S. Computer Science, State University, 2019

SKILLS
Python, PostgreSQL, Docker, Kubernetes, AWS, Terraform, Git, CI/CD

CERTIFICATIONS
AWS Certified Solutions Architect
"""

SAMPLE_JD_TEXT = """Senior Backend Engineer - Platform

We are looking for a Senior Backend Engineer to join our platform team.

Responsibilities:
- Design and build scalable backend services in Python
- Own the reliability of our data pipelines and REST APIs
- Mentor engineers and drive technical direction

Required qualifications:
- 5+ years of backend engineering experience
- Expert in Python, PostgreSQL and REST APIs
- Experience with Docker and Kubernetes
- Bachelor's degree in Computer Science or related field

Preferred qualifications:
- Experience with Kafka, Airflow, or Spark
- AWS experience and hands-on Terraform knowledge
- Experience mentoring engineers

About the role:
Indianapolis based hybrid role. Great benefits.
"""
