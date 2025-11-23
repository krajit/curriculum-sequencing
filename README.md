# Curriculum Sequencing (Retention-Weighted)

This repository reproduces experiments from the manuscript by Ajit Kumar.

Files:
- course_selector.py
- connectivityMatrix.csv (16-course example)
- requirements.txt
- LICENSE

Uploaded draft PDF (used as reference in manuscript):
file:///mnt/data/A_mathematical_model_to_maximize_information_retention_by_students_in_college_courses.pdf

Quickstart:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 course_selector.py --csv connectivityMatrix.csv --nc 2 --alpha 0.5 --enumeration_limit 50000
