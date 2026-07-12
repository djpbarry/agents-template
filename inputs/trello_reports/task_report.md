# Trello Board Analysis - Specific Task

## Objective

Analyze a Trello board JSON export and generate a Python script that computes key statistics and creates visualizations to understand team workload and board organization.

## Input

Trello board exported as JSON file containing: cards, lists, members, labels, checklists, custom fields, and activity history.

## Analysis Requirements

### 1. Compute Key Statistics
Extract and print the following metrics:
- **Total cards**: Count of all cards in the board
- **Cards per list**: Count of cards in each list (tabular format)
- **Cards per member**: Count of cards assigned to each team member
- **Overdue cards**: Count and list of cards with past due dates
- **Completion rate**: Percentage of checklists marked complete across the board

### 2. Create Three Visualizations (saved as PNG files)

1. **Cards by List** (Bar Chart)
   - X-axis: List names
   - Y-axis: Card count
   - Shows distribution of work across board lists
   - File: `cards_by_list.png`

2. **Workload by Member** (Bar Chart)
   - X-axis: Team member names
   - Y-axis: Number of cards assigned
   - Shows workload distribution across team
   - File: `workload_by_member.png`

3. **Card Status Distribution** (Pie Chart)
   - Slices: Count of cards by list (or status if determinable from list names)
   - Shows percentage breakdown of board focus areas
   - File: `card_status_distribution.png`

### 3. Identify Data Gaps

After analysis, print 2-3 specific suggestions for additional data/fields that would improve future analysis:
- Examples: due date compliance tracking, priority levels, effort estimates, blockers/dependencies, sprint assignments, etc.
- Format: Bulleted list with brief explanation of why each would be useful

## Output Requirements

The script must:
- Load and parse the Trello JSON export (auto-detect filename in the data directory)
- Compute all statistics listed in section 1
- Print statistics to console in a readable tabular format
- Generate 3 PNG visualization files (using matplotlib)
- Print data gap analysis suggestions to console
- Save visualizations to the current working directory with specified filenames
- Handle missing or empty data gracefully (e.g., members with no cards, lists with no due dates)

## Success Criteria

✅ Script runs without errors on the provided Trello JSON export  
✅ All 5 statistics are computed and printed  
✅ Three PNG visualization files are created and saved  
✅ Visualizations are properly labeled (titles, axis labels, legends)  
✅ Data gap analysis provides 2-3 specific, actionable suggestions  
✅ Code is clean and minimal (no unnecessary utilities or visualizations)  
✅ One-line docstrings for all functions  
