import os
import re
import openai
import pdfplumber
import spacy
import textwrap
from docx import Document
from collections import Counter
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, render_template
from geotext import GeoText
from werkzeug.utils import secure_filename


nlp = spacy.load("en_core_web_md")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

openai.api_key = "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"

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_candidate_phone_number(text):
    phone_regex = re.compile(r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}')
    match = phone_regex.search(text)
    return match.group(0) if match else "Not Provided"

def extract_candidate_email_address(text):
    email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    match = email_regex.search(text)
    return match.group(0) if match else "Not Provided"

def extract_highest_education_level(text):
    education_levels = [
        "Doctorate", "Ph.D.", "PhD", "Ed.D.", "EdD",
        "Master's", "Masters", "M.Sc.", "MSc", "M.A.", "MA", "M.Eng.", "MEng", "MBA",
        "Bachelor's", "Bachelors", "B.Sc.", "BSc", "B.A.", "BA", "B.Eng.", "BEng",
        "Associate's", "Associates", "A.A.", "AA", "A.S.", "AS",
        "High School Diploma", "GED"
    ]

    highest_level_found = "Not Provided"
    for level in education_levels:
        if level in text:
            highest_level_found = level
            break

    return highest_level_found


def read_file(file):
    file_extension = os.path.splitext(file.filename)[1].lower()
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOADS_DIR_ABS, filename)

    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Please use a .pdf or .docx file.")


def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
        text = "\n".join(pages)
    return text


def extract_text_from_docx(file_path):
    print(f'Opening file at: {file_path}')  # Add this line
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)


def match_resume_and_job_description(resume_doc, job_description_doc, candidate_name, work_mode, planned_pto):
    resume_nlp = nlp(resume_doc)
    job_description_nlp = nlp(job_description_doc)

    similarity = resume_nlp.similarity(job_description_nlp)
    similarity_percent = round(similarity * 100, 2)  # convert to percentage and round to 2 decimal places

    phone_number = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', resume_doc)
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_doc)
    candidate_info = {
        'candidate_name': candidate_name,
        'work_mode': work_mode,
        'candidate_phone': phone_number[0] if phone_number else '',
        'candidate_email': email[0] if email else '',
        'planned_pto': planned_pto,
        'agreed_salary': '$54.00/hr',
        'candidate_location': extract_candidate_location(resume_doc),
        'education': extract_candidate_education(resume_doc),
        'similarity': similarity_percent
    }
    return candidate_info


def generate_candidate_summary(resume_text, job_description_text):
    prompt = f"Create a summary for a resume that demonstrates why the candidate is a good fit for the following job description:\n\nResume:\n{resume_text}\n\nJob Description:\n{job_description_text}\n\nSummary:"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    summary = response.choices[0].text.strip()
    return summary


def extract_relevant_experience(resume_text, job_description_text):
    prompt = f"Given the following resume and job description, list the relevant work experience and total number of months of relevant experience:\n\nResume:\n{resume_text}\n\nJob Description:\n{job_description_text}\n\nRelevant Experience:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    result = response.choices[0].text.strip()

    # Find the total number of months in the result string
    months = re.findall(r"(\d+) months", result)
    total_months = sum(map(int, months)) if months else 0

    return result, total_months



def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return " ".join(tokens)

def create_template(candidate_info, resume_text):
    template = """
<h3>Resume of: {candidate_name}</h3>
<p><strong>Candidate Location:</strong> {candidate_location}</p>
<p><strong>Onsite/Hybrid/Remote:</strong> {work_mode}</p>
<p><strong>Candidate Phone Number:</strong> {phone_number}</p>
<p><strong>Candidate Email:</strong> {email}</p>
<p><strong>Highest Level of Education Completed:</strong> {education}</p>

<h3>Resume:</h3>
<p>{resume_text}</p>
"""
    total_relevant_experience_months = candidate_info['total_relevant_experience_months']

    # Here, we split the relevant_experience string by '-' and add "<br>-" before each item except the first one.
    relevant_experience_items = candidate_info['relevant_experience'].split('-')
    relevant_experience_formatted = "<br>-".join(relevant_experience_items)

    return template.format(candidate_name=candidate_info['candidate_name'],
                           candidate_location=candidate_info['candidate_location'],
                           work_mode=candidate_info['work_mode'],
                           phone_number=candidate_info['candidate_phone'],
                           email=candidate_info['candidate_email'],
                           education=candidate_info['education'],
                           resume_text=resume_text) + "<br><strong>Relevant Experience:</strong><br>- " + relevant_experience_formatted + "<br><strong>Total Months of Relevant Experience:</strong> " + str(total_relevant_experience_months)




# Flask routes code remains unchanged.
from geotext import GeoText

def extract_candidate_location(text):
    places = GeoText(text)
    cities = list(places.cities)
    if cities:
        return cities[0]
    return "Not Provided"

def extract_candidate_education(text):
    education_levels = ["High School Diploma", "Associate's Degree", "Bachelor's Degree",
                        "Master's Degree", "Ph.D.", "MBA", "Doctorate"]
    for level in education_levels:
        if level in text:
            return level
    return "Not Provided"

@app.route('/')
def index():
    return render_template('index.html')

UPLOADS_DIR = 'uploads'
UPLOADS_DIR_ABS = os.path.abspath(UPLOADS_DIR)

@app.route('/convert', methods=['POST'])
def convert():
    max_tokens = 4500
    if 'job-description-file' not in request.files or 'resume-files' not in request.files:
        flash('No file part')
        return redirect(request.url)

    job_description_file = request.files['job-description-file']

    # Ensure filenames don't have leading/trailing spaces
    job_description_filename = secure_filename(job_description_file.filename.strip())

    # Use absolute path
    job_description_path = os.path.join(UPLOADS_DIR_ABS, job_description_filename)

    # Save the job description file
    job_description_file.save(job_description_path)

    resume_files = request.files.getlist('resume-files')
    results = []

    experience_unit = request.form.get("experience-unit")
    candidate_name = request.form.get("candidate-name")
    work_mode = request.form.get("work-mode")
    planned_pto = request.form.get("planned-pto")

    job_description_text = read_file(job_description_file)

    for resume_file in resume_files:
        # Ensure filenames don't have leading/trailing spaces
        resume_filename = secure_filename(resume_file.filename.strip())

        # Use absolute path
        resume_file_path = os.path.join(UPLOADS_DIR_ABS, resume_filename)

        # Save the resume file
        resume_file.save(resume_file_path)

        resume_text = read_file(resume_file)


        relevant_experience, total_months = extract_relevant_experience(resume_text, job_description_text)
        truncated_resume_text = truncate_text(resume_text, max_tokens // 2)
        truncated_job_description_text = truncate_text(job_description_text, max_tokens // 2)

        candidate_info = match_resume_and_job_description(resume_text, job_description_text,
                                                          candidate_name, work_mode, planned_pto)

        candidate_info['candidate_phone'] = extract_candidate_phone_number(resume_text)
        candidate_info['candidate_email'] = extract_candidate_email_address(resume_text)
        candidate_info['education_level'] = extract_highest_education_level(resume_text)
        candidate_info['relevant_experience'] = relevant_experience
        candidate_info['total_relevant_experience_months'] = total_months
        candidate_info['resume_text'] = resume_text
        candidate_info['resume_summary'] = generate_candidate_summary(truncated_resume_text,
                                                                      truncated_job_description_text)
        candidate_info['experience_unit'] = experience_unit
        formatted_resume = create_template(candidate_info, resume_text)
        summary = candidate_info['resume_summary']

        result = {"summary": summary,
                  "formatted_resume": formatted_resume,
                  "relevant_experience": relevant_experience,
                  "similarity_score": candidate_info['similarity']}
        results.append(result)

    sorted_results = sorted(results, key=lambda r: r['similarity_score'], reverse=True)

    return jsonify(sorted_results), 200


@app.route("/result")
def result():
    return render_template("result.html")


if __name__ == '__main__':
    app.run(debug=True)
