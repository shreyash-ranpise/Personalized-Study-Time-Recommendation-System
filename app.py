from flask import Flask, render_template, request, jsonify
from src.data_loader import load_student_data
from src.models import train_models, recommend_study_plan
import math

app = Flask(__name__)

# Load and train models once at startup
print("Loading dataset and training models...")
df = load_student_data()
models, metrics = train_models(df)
print("Models trained successfully!")

SUBJECT_OPTIONS = [
    "Database Management Systems",
    "Data Science",
    "Probability and Statistics",
    "Embedded System",
    "Open Elective II",
    "Environmental Studies",
]

def difficulty_to_text(diff: float) -> str:
    """Convert difficulty number to text label."""
    diff_int = int(round(diff))
    if diff_int == 1:
        return "very easy"
    elif diff_int == 2:
        return "easy"
    elif diff_int == 3:
        return "medium"
    elif diff_int == 4:
        return "hard"
    elif diff_int == 5:
        return "very hard"
    else:
        return "medium"

def hours_to_str(hours: float) -> str:
    """Convert hours (float) to 'X hrs YY min' format."""
    total_minutes = int(round(hours * 60))
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{h:d} hrs {m:02d} min"

@app.route('/')
def index():
    """Main page with input form."""
    return render_template('index.html', subjects=SUBJECT_OPTIONS)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Process form data and return recommendations."""
    try:
        # Parse form data
        sleep_hours = float(request.form.get('sleep_hours', 7.0))
        college_start = request.form.get('college_start', '09:00')
        college_end = request.form.get('college_end', '15:00')
        travel_time = float(request.form.get('travel_time', 1.0))
        screen_time = float(request.form.get('screen_time', 3.0))
        
        # Parse college times
        start_parts = college_start.split(':')
        end_parts = college_end.split(':')
        start_hours = int(start_parts[0]) + int(start_parts[1]) / 60.0
        end_hours = int(end_parts[0]) + int(end_parts[1]) / 60.0
        class_hours = end_hours - start_hours
        
        if class_hours <= 0 or class_hours > 24:
            return jsonify({'error': 'Invalid college timings. End time must be after start time.'}), 400
        
        # Academic profile
        academic_choice = request.form.get('academic_choice', '1')
        if academic_choice == '1':
            cgpa = float(request.form.get('cgpa', 7.0))
            previous_marks = cgpa * 10.0
            backlogs = 0.0
        else:
            backlogs = float(request.form.get('backlogs', 0.0))
            previous_marks = 70.0
        
        # Get subjects and difficulties
        subjects = []
        for subj in SUBJECT_OPTIONS:
            difficulty_key = f'difficulty_{subj.replace(" ", "_").lower()}'
            difficulty = request.form.get(difficulty_key)
            if difficulty:
                subjects.append({
                    'name': subj,
                    'difficulty': float(difficulty)
                })
        
        if not subjects:
            return jsonify({'error': 'Please enter difficulty for at least one subject.'}), 400
        
        # Validate total time
        total_time = sleep_hours + class_hours + travel_time + screen_time
        if total_time > 24.0:
            return jsonify({
                'error': f'Total routine time ({total_time:.2f} hours) exceeds 24 hours. Please enter realistic values.'
            }), 400
        
        remaining_hours = max(0.0, 24.0 - total_time)
        
        # Prepare inputs for model
        avg_difficulty = sum(s['difficulty'] for s in subjects) / len(subjects)
        user_inputs = {
            'sleep_hours': sleep_hours,
            'class_hours': class_hours,
            'travel_time': travel_time,
            'screen_time': screen_time,
            'subject_difficulty': avg_difficulty,
            'previous_marks': previous_marks,
            'backlogs': backlogs,
            'college_start': start_hours,
        }
        
        # Get base recommendation
        base_hours, slot = recommend_study_plan(models, user_inputs, max_hours_per_day=remaining_hours)
        
        # CGPA-based adjustment
        cgpa = previous_marks / 10.0
        cgpa_adjustment = 0.0
        if cgpa <= 5.0:
            cgpa_adjustment = 1.0
        elif cgpa <= 7.0:
            cgpa_adjustment = 0.5
        elif cgpa >= 9.0:
            cgpa_adjustment = -0.75
        elif cgpa >= 8.5:
            cgpa_adjustment = -0.5
        
        # Calculate total study hours
        total_weight = sum(s['difficulty'] for s in subjects)
        if total_weight == 0:
            total_weight = len(subjects)
        
        total_study_hours = base_hours + cgpa_adjustment
        total_study_hours = max(0.5, min(total_study_hours, remaining_hours))
        
        # Distribute across subjects
        subject_allocations = []
        for subj in subjects:
            weight = subj['difficulty']
            if total_weight > 0:
                allocated_hours = (weight / total_weight) * total_study_hours
            else:
                allocated_hours = total_study_hours / len(subjects)
            allocated_hours = max(0.25, allocated_hours)
            subject_allocations.append({
                'name': subj['name'],
                'difficulty': subj['difficulty'],
                'hours': allocated_hours
            })
        
        # Normalize to fit remaining hours
        actual_total = sum(a['hours'] for a in subject_allocations)
        if actual_total > remaining_hours:
            scale_factor = remaining_hours / actual_total
            for a in subject_allocations:
                a['hours'] *= scale_factor
        
        # Adjust slot if college is in morning
        if 6.0 <= start_hours <= 12.0 and slot == "Morning":
            slot = "Evening"
        
        # Format results
        results = {
            'base_time': round(total_time, 2),
            'remaining_hours': round(remaining_hours, 2),
            'total_study_hours': round(sum(a['hours'] for a in subject_allocations), 2),
            'best_time_slot': slot,
            'subjects': []
        }
        
        # Find max length for alignment
        max_before_arrow = 0
        for alloc in subject_allocations:
            diff_text = difficulty_to_text(alloc['difficulty'])
            before_arrow = f"{alloc['name']} (Difficulty: {diff_text})"
            max_before_arrow = max(max_before_arrow, len(before_arrow))
        
        for alloc in subject_allocations:
            diff_text = difficulty_to_text(alloc['difficulty'])
            hours_str = hours_to_str(alloc['hours'])
            before_arrow = f"{alloc['name']} (Difficulty: {diff_text})"
            padded_before = before_arrow.ljust(max_before_arrow)
            
            results['subjects'].append({
                'name': alloc['name'],
                'difficulty_text': diff_text,
                'hours': alloc['hours'],
                'hours_str': hours_str,
                'formatted_line': f"{padded_before} → {hours_str}"
            })
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
