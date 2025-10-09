import csv
from collections import Counter

log_file = 'violation_log.csv'

def generate_report():
    try:
        with open(log_file, mode='r') as file:
            reader = csv.reader(file)
            # Skip header if present, or read all rows
            violations = list(reader)

        violation_types = [row[1] for row in violations]  # violation type at index 1
        counts = Counter(violation_types)

        print("Violation Report Summary:")
        for violation, count in counts.items():
            print(f"{violation}: {count}")

    except FileNotFoundError:
        print(f"No log file found at {log_file}. Run detection first to generate logs.")

if __name__ == "__main__":
    generate_report()
