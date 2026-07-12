# Trello Board Analysis - Specific Task

## Objective

Analyze a Trello board JSON export and generate a Python script that computes key statistics and creates visualizations 
to understand team workload and board organization. Identify gaps in data collection and suggest improvements.

## Input

Trello board exported as JSON file containing: cards, lists, members, labels, checklists, custom fields, and activity history.

## Analysis Requirements

### 1. Suggest Key Statistics
Analyse the Trello board JSON export and suggest key statistics to understand team workload and board organization. 
Consider the following metrics:
- **Total cards**: Count of all cards in the board
- **Cards per member**: Count of cards assigned to each team member
- **Time to archive**: Average time taken for cards to be archived after creation
- **Time in list**: Average time cards spend in each list before being archived
- **Time between updates**: Average time between card updates
- **Breakdown across different labs**: Total number of cards per lab in each list

### 2. Create Three Visualizations (saved as PNG files)
Create visualisations to illustrate at least 3 of the statistics suggested in section 1. 

### 3. Identify Data Gaps

After analysis, print 2-3 specific suggestions for additional data/fields that would improve future analysis:
- Examples: due date compliance tracking, priority levels, effort estimates, blockers/dependencies, sprint assignments, etc.
- Format: Bulleted list with a brief explanation of why each would be useful

## Output Requirements

The script must:
- Load and parse the Trello JSON export (auto-detect filename in the data directory)
- Compute statistics suggested in section 1
- Print statistics to console in a readable tabular format
- Generate the visualizations suggested in section 2
- Print data gap analysis suggestions to console
- Save visualizations to the current working directory with specified filenames
- Handle missing or empty data gracefully (e.g., members with no cards, lists with no due dates)

## Success Criteria

✅ Script runs without errors on the provided Trello JSON export  
✅ At least 5 statistics are computed and printed  
✅ At least three PNG visualization files are created and saved  
✅ Visualisations are properly labeled (titles, axis labels, legends)   
✅ Code is clean and minimal (no unnecessary utilities or visualizations)  
✅ One-line docstrings for all functions  
