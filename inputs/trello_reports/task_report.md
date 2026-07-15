# Trello Board Analysis - Exploratory Task

## Objective

Analyze a Trello board JSON export to explore and answer key questions about team workflow, workload distribution, and process health. Generate metrics and visualizations that provide insight into how work flows through the board and where improvements could be made.

## Input

Trello board exported as JSON file containing: cards, lists, members, labels, checklists, custom fields, and activity history.
Of particular interest are the custom fields, particularly "Lab (Name)", "Source" and "Lead".

## Guiding Questions for Analysis

Your analysis should help answer these exploratory questions:

1. **Workflow Bottlenecks**: Where do cards get stuck? Which lists do cards spend the most time in? Are there significant delays between creation and completion?

2. **Team Workload Distribution**: How is work distributed across team members? Are some members overloaded while others have capacity? Which members handle which types of work?

3. **Velocity & Timing Patterns**: How long does it typically take for cards to move through the workflow? Is the process getting faster or slower? Are there patterns in how often cards are updated?

4. **Process Health**: How many cards are in progress vs. completed? Are there inactive cards that should be archived? How is the board being used—is activity even or bursty?

## Analysis Requirements

### 1. Suggest Key metrics
Analyse the Trello board JSON export and suggest **5-7 key metrics** that help answer the guiding questions above. Choose metrics that are:
- Computable from the available data
- Relevant to at least one of the 4 guiding questions
- More insightful than just raw counts (consider timing, distribution, and patterns)

Examples might include: cards per member, time-to-archive, time-in-list, update frequency, bottleneck lists, velocity trends, etc.
Pay particular attention here to the custom fields and labels used in the Trello board – any insights derived from these are of particular interest.
Consider other metrics that might typically be included in an analysis of Trello board activity, or project management in general.

### 2. Create Visualizations (3+ PNG files)
Create **at least 3 visualizations** that illustrate the metrics chosen in section 1. Pick visualizations that help answer the guiding questions:
- Time-series or trend plots if timing data is available
- Heatmaps to show patterns across dimensions
- scatter plots to show data distribution and variability
- Avoid bar charts and pie charts

### 3. Identify Data Gaps

After analysis, print **2-3 specific suggestions** for additional data/fields that would improve future analysis:
- Focus on what data would help answer the guiding questions more clearly
- Examples: due date compliance, priority levels, effort estimates, blockers/dependencies, sprint assignments, etc.
- Format: Bulleted list with a brief explanation of why each would be useful

## Output Requirements

The script must:
- Load and parse the Trello JSON export (auto-detect filename in the data directory)
- Compute metrics suggested in section 1
- Print metrics to console in a readable tabular format
- Generate the visualizations suggested in section 2
- Print data gap analysis suggestions to console
- Save visualizations to the current working directory with specified filenames
- Handle missing or empty data gracefully (e.g., members with no cards, lists with no due dates)

## Success Criteria

✅ Script runs without errors on the provided Trello JSON export  
✅ At least five metrics are computed and printed  
✅ At least three PNG visualization files are created and saved  
✅ Visualisations are properly labeled (titles, axis labels, legends)   
✅ Code is clean and minimal (no unnecessary utilities or visualizations)  
✅ One-line docstrings for all functions  
