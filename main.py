from typing import Dict, Tuple, List
from src.data_loader import load_student_data
from src.models import train_models, recommend_study_plan

STAR_LINE = "*******************************************************************************"


def print_banner(title: str) -> None:
    """
    Prints a single-line banner like
    *************** TITLE ***************
    with total length matching STAR_LINE.
    """
    text = f" {title} "
    print("\n" + f"{text:*^{len(STAR_LINE)}}")

SUBJECT_OPTIONS = [
    "Database Management Systems",
    "Data Science",
    "Probability and Statistics",
    "Embedded System",
    "Open Elective II",
    "Environmental Studies",
]

def collect_user_inputs() -> Tuple[Dict[str, float], List[Dict[str, float | str]]]:
    """
    Collects routine and academic details from the user via console input.
    """
    print("\nEnter your details to get a personalized study time recommendation.")
    print("\nPress Enter to use the example value shown in brackets.\n")

    def ask_float(
        prompt: str,
        default: float,
        max_value: float | None = None,
        min_value: float | None = None,
        default_label: str | None = None,
        is_time: bool = False,
    ) -> float:
        while True:
            label = default_label if default_label is not None else str(default)
            raw = input(f"{prompt} [{label}]: ").strip()
            if not raw:
                value = default
            else:
                # Try to parse plain numeric first
                try:
                    value = float(raw)
                except ValueError:
                    if is_time:
                        # Try to parse formats like "7 hrs 30 min" or "7h 30m"
                        import re

                        lower = raw.lower()
                        pattern = r"^\s*(\d+)\s*(?:h|hr|hrs|hour|hours)?\s*(\d+)?\s*(?:m|min|mins|minute|minutes)?\s*$"
                        match = re.match(pattern, lower)
                        if match:
                            hours_part = int(match.group(1))
                            minutes_part = int(match.group(2)) if match.group(2) is not None else 0
                            value = hours_part + minutes_part / 60.0
                        else:
                            print("\nPlease enter time as a number of hours (e.g., 7.5) or in the form '7 hrs 30 min'.")
                            continue
                    else:
                        print("\nPlease enter a valid number.")
                        continue

            if min_value is not None and value < min_value:
                print(f"\nValue cannot be less than {min_value}. Please enter a valid number.")
                continue

            if max_value is not None and value > max_value:
                print(f"\nValue cannot be greater than {max_value}. Please enter a valid number.")
                continue

            return value

    def parse_time_hhmm(prompt: str, default_str: str) -> float:
        """
        Ask the user for a time of day in HH:MM (24-hour) format and return hours as float.
        """
        while True:
            raw = input(f"{prompt} [{default_str}]: ").strip()
            if not raw:
                raw = default_str
            try:
                parts = raw.split(":")
                if len(parts) != 2:
                    raise ValueError
                h = int(parts[0])
                m = int(parts[1])
                if not (0 <= h < 24 and 0 <= m < 60):
                    raise ValueError
                return h + m / 60.0
            except ValueError:
                print("\nPlease enter time in HH:MM format, 24-hour clock (e.g., 09:00 or 14:30).")

    # Loop until the total daily time is valid (<= 24 hours)
    while True:
        print_banner("INSERT DATA")

        # Helper to build a label like "7 hrs 30 min" from a float value in hours
        def hours_label(value: float) -> str:
            total_minutes = int(round(value * 60))
            h = total_minutes // 60
            m = total_minutes % 60
            return f"{h} hrs {m:02d} min"

        # Sleep hours (direct hours input)
        inputs = {
            "sleep_hours": ask_float(
                "\nAverage sleep hours per day",
                7.0,
                min_value=0.0,
                max_value=24.0,
                default_label=hours_label(7.0),
                is_time=True,
            ),
        }

        # College time: ask start and end, then compute class_hours
        print("\nEnter your college timing (24-hour format):")
        while True:
            start_hours = parse_time_hhmm("  College start time (HH:MM)", "09:00")
            end_hours = parse_time_hhmm("  College end time   (HH:MM)", "15:00")

            class_duration = end_hours - start_hours
            if class_duration <= 0 or class_duration > 24:
                print(
                    "\nEnd time must be after start time and within the same day. "
                    "Please enter realistic college timings."
                )
                continue

            inputs["class_hours"] = class_duration
            # Store college start time separately (in hours) for rule-based slot adjustment later
            inputs["college_start"] = start_hours
            break

        # Travel, screen time, and difficulty
        inputs["travel_time"] = ask_float(
            "\nDaily travel time (hours)",
            1.0,
            min_value=0.0,
            max_value=24.0,
            default_label=hours_label(1.0),
            is_time=True,
        )
        inputs["screen_time"] = ask_float(
            "\nDaily non-study screen time (hours)",
            3.0,
            min_value=0.0,
            max_value=24.0,
            default_label=hours_label(3.0),
            is_time=True,
        )

        # Either ask for previous year CGPA OR number of backlogs
        print(
            "\nAcademic profile: you can either provide your previous year CGPA "
            "or the number of backlogs."
        )
        print("  1. Enter previous year CGPA (0–10)")
        print("  2. Enter number of backlogs (arrears)")

        choice_academic = ""
        while choice_academic not in {"1", "2"}:
            choice_academic = input("\nChoose option 1 or 2 [1]: ").strip() or "1"
            if choice_academic not in {"1", "2"}:
                print("\nPlease enter 1 or 2.")

        if choice_academic == "1":
            # Ask for CGPA and convert internally to a 0–100 scale to match the training data
            cgpa = ask_float("\nPrevious CGPA (0-10)", 7.0, min_value=0.0, max_value=10.0)
            inputs["previous_marks"] = cgpa * 10.0
            # Assume no backlogs if CGPA is provided
            inputs["backlogs"] = 0.0
        else:
            # Ask for number of backlogs (integer)
            while True:
                backlogs = ask_float(
                    "\nNumber of backlogs (arrears)", 0.0, min_value=0.0, max_value=20.0
                )
                if float(backlogs).is_integer():
                    inputs["backlogs"] = float(backlogs)
                    break
                print("\nPlease enter a whole number (integer) for backlogs.")
            # Use a neutral default for previous_marks when only backlogs are given
            inputs["previous_marks"] = 70.0

        # Ask for subjects and their difficulties
        print("\nEnter difficulty level for each subject:")
        subjects: List[Dict[str, float | str]] = []
        for idx, subject_name in enumerate(SUBJECT_OPTIONS, start=1):
            print(f"\nSubject {idx}: {subject_name}")
            difficulty = ask_float(
                f"Enter difficulty (1=easy, 5=very hard)",
                3.0,
                min_value=1.0,
                max_value=5.0,
            )
            subjects.append({"name": subject_name, "difficulty": difficulty})

        total_time = (
            inputs["sleep_hours"]
            + inputs["class_hours"]
            + inputs["travel_time"]
            + inputs["screen_time"]
        )

        if total_time > 24.0:
            print(
                f"\nThe total of sleep hours + class hours + travel time + non-study screen time "
                f"is {total_time:.2f} hours, which exceeds 24 hours in a day."
            )
            print("Please enter realistic values so that the total does not exceed 24 hours.\n")
            continue

        return inputs, subjects

def main() -> None:
    print_banner("Personalized Study Time Recommendation System")

    df = load_student_data()
    models, metrics = train_models(df)

    print("\nModel trained on sample dataset.")
    print(f"- Study hours MAE: {metrics['hours_mae']:.3f}")
    print(f"- Study hours RMSE: {metrics['hours_rmse']:.3f}")
    print(f"- Time slot accuracy: {metrics['slot_accuracy']:.3f} on {metrics['test_samples']} test samples\n")

    while True:
        user_inputs, subjects = collect_user_inputs()

        if not subjects:
            print("\nNo subjects entered. Please enter at least one subject.")
            continue

        # Ensure recommended study time fits into the remaining hours of the day
        base_time = (
            user_inputs["sleep_hours"]
            + user_inputs["class_hours"]
            + user_inputs["travel_time"]
            + user_inputs["screen_time"]
        )
        remaining_hours = max(0.0, 24.0 - base_time)

        # Get base recommendation from model (using average difficulty for the model call)
        avg_difficulty = sum(s["difficulty"] for s in subjects) / len(subjects)
        user_inputs["subject_difficulty"] = avg_difficulty
        base_hours, slot = recommend_study_plan(models, user_inputs, max_hours_per_day=remaining_hours)

        # Rule-based adjustment based on CGPA
        cgpa = user_inputs["previous_marks"] / 10.0  # previous_marks is kept as percentage internally
        
        # Base adjustment from CGPA
        cgpa_adjustment = 0.0
        if cgpa <= 5.0:
            cgpa_adjustment = 1.0  # Low CGPA → need more study time overall
        elif cgpa <= 7.0:
            cgpa_adjustment = 0.5
        elif cgpa >= 9.0:
            cgpa_adjustment = -0.75  # High CGPA → can study less
        elif cgpa >= 8.5:
            cgpa_adjustment = -0.5

        # Calculate total study hours needed based on all subjects
        # Distribute time proportionally based on difficulty
        total_weight = sum(s["difficulty"] for s in subjects)
        if total_weight == 0:
            total_weight = len(subjects)  # Fallback if all difficulties are 0
        
        # Base total hours: model prediction + CGPA adjustment
        total_study_hours = base_hours + cgpa_adjustment
        total_study_hours = max(0.5, min(total_study_hours, remaining_hours))

        # Distribute study time across subjects based on difficulty
        subject_allocations = []
        for subj in subjects:
            # Weight based on difficulty (harder = more time)
            weight = subj["difficulty"]
            if total_weight > 0:
                allocated_hours = (weight / total_weight) * total_study_hours
            else:
                allocated_hours = total_study_hours / len(subjects)
            
            # Ensure minimum 0.25 hours per subject
            allocated_hours = max(0.25, allocated_hours)
            subject_allocations.append({
                "name": subj["name"],
                "difficulty": subj["difficulty"],
                "hours": allocated_hours
            })

        # Normalize to ensure total doesn't exceed remaining hours
        actual_total = sum(a["hours"] for a in subject_allocations)
        if actual_total > remaining_hours:
            scale_factor = remaining_hours / actual_total
            for a in subject_allocations:
                a["hours"] *= scale_factor

        # Rule-based adjustment: if college is in the morning, avoid suggesting Morning for study
        college_start = user_inputs.get("college_start", None)
        adjusted_slot = slot
        if college_start is not None and 6.0 <= college_start <= 12.0 and slot == "Morning":
            adjusted_slot = "Evening"

        print_banner("Daily Time Summary")
        print(f"\nTotal routine time (sleep + class + travel + screen): {base_time:.2f} hours")
        print(f"Remaining available hours in the day: {remaining_hours:.2f} hours")

        print_banner("Recommendation")
        print(f"\nTotal recommended study time: {sum(a['hours'] for a in subject_allocations):.2f} hours")
        print(f"\nSuggested best time to study: {adjusted_slot}")
        # Helper function to convert difficulty number to text
        def difficulty_to_text(diff: float) -> str:
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
                return "medium"  # fallback

        print("\nStudy time breakdown by subject:")
        # Calculate the maximum length needed before the arrow to align it
        max_before_arrow = 0
        for i, alloc in enumerate(subject_allocations, 1):
            diff_text = difficulty_to_text(alloc["difficulty"])
            before_arrow = f"  {i}. {alloc['name']} (Difficulty: {diff_text})"
            max_before_arrow = max(max_before_arrow, len(before_arrow))
        
        for i, alloc in enumerate(subject_allocations, 1):
            total_minutes = int(round(alloc["hours"] * 60))
            h = total_minutes // 60
            m = total_minutes % 60
            hours_str = f"{h:d} hrs {m:02d} min"
            diff_text = difficulty_to_text(alloc["difficulty"])
            before_arrow = f"  {i}. {alloc['name']} (Difficulty: {diff_text})"
            # Pad to align the arrow
            padded_before = before_arrow.ljust(max_before_arrow)
            print(f"{padded_before} → {hours_str}")
        
        print_banner("NOTE")
        print("\nThese recommendations are based on a small sample dataset and simple models,")
        print("\nso they are approximate and should be adjusted to your real-life comfort and results.")
        print("\n*******************************************************************************")


        # Ask user if they want another recommendation
        again = input("\nDo you want to get another recommendation? (y/n) [n]: ").strip().lower()
        if again not in {"y", "yes"}:
            print("\nThank you for using the Personalized Study Time Recommendation System. Goodbye!")
            print("\n*******************************************************************************")
            break


if __name__ == "__main__":
    main()

