import pandas as pd
import matplotlib.pyplot as plt

# File path
logfile = 'violation_log.csv'

# Load CSV with manual column headers since file lacks them
df = pd.read_csv(logfile, header=None, names=["Timestamp", "ViolationType", "X1", "Y1", "X2", "Y2"])

# Convert Timestamp column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract date only for grouping
df['Date'] = df['Timestamp'].dt.date

# Group by Date and ViolationType for counts
summary = df.groupby(['Date', 'ViolationType']).size().unstack(fill_value=0)

# Choose one of the following chart styles by uncommenting the desired block:

# 1. Bar Chart (default)
summary.plot(kind='bar', figsize=(10, 6))
plt.title('Helmet and Vest Violations Over Time - Bar Chart')
plt.ylabel('Number of Violations')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Line Plot
# summary.plot(kind='line', marker='o', figsize=(10, 6))
# plt.title('Helmet and Vest Violations Over Time - Line Plot')
# plt.ylabel('Number of Violations')
# plt.xlabel('Date')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# 3. Stacked Bar Chart
# summary.plot(kind='bar', stacked=True, figsize=(10, 6))
# plt.title('Helmet and Vest Violations Over Time - Stacked Bar Chart')
# plt.ylabel('Number of Violations')
# plt.xlabel('Date')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# 4. Single Violation Type Plot (Helmet Missing example)
# summary['Helmet Missing'].plot(kind='bar', figsize=(10, 6))
# plt.title('Helmet Missing Violations Over Time')
# plt.ylabel('Number of Violations')
# plt.xlabel('Date')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
