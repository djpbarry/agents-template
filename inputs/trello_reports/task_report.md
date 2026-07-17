# Trello Board Analysis - Exploratory Task

## Objective

Analyze a Trello board JSON export to explore and answer key questions about team workflow, workload distribution, and process health. Generate metrics and visualizations that provide insight into how work flows through the board and where improvements could be made.

## Input

Trello board exported as JSON file containing: cards, lists, members, labels, checklists, custom fields, and activity history.
Of particular interest are the custom fields, particularly "Lab (Name)", "Source" and "Lead".

## Guiding Questions for Analysis

Your analysis should help answer these exploratory questions:

1. **Workflow Bottlenecks**: Where do cards get stuck? Which lists do cards spend the most time in? Are there significant delays between creation and completion? Do cards from certain labs go stale more often than others?

2. **Team Workload Distribution**: How is work distributed across team members? Are some members overloaded while others have capacity? Which members handle which types of work?

3. **Velocity & Timing Patterns**: How long does it typically take for cards to move through the workflow? Is the process getting faster or slower? Are there patterns in how often cards are updated?

4. **Process Health**: How many cards are in progress vs. completed? Are there inactive cards that should be archived? How is the board being used—is activity even or bursty?

5. **Client Relationship Management**: Do specific labs or users have a preference for specific team members? Do people in the same lab open projects with multiple team members?

6. **Lab workload distribution**: Which labs do the team spend most time working with? How has this evolved over time? Are projects with certain labs more productive than others?

## Analysis Requirements

### 1. Suggest Key metrics
Analyse the Trello board JSON export and suggest **5-7 key metrics** that help answer the guiding questions above. Choose metrics that are:
- Computable from the available data
- Relevant to at least one of the 4 guiding questions
- More insightful than just raw counts (consider timing, distribution, and patterns)
- Examine carefully the plot examples in the MatplotLib and Seaborn galleries and consider whether these might inform your choice of metrics:
  - https://github.com/matplotlib/matplotlib/tree/main/galleries
  - https://github.com/mwaskom/seaborn/tree/master/examples
- Pay particular attention here to the custom fields and labels used in the Trello board – any insights derived from these are of particular interest.
- Consider other metrics that might typically be included in an analysis of Trello board activity, or project management in general.

### 2. Create Visualizations (PNG files)
Create **at least five visualizations** that illustrate the metrics chosen in section 1. Pick visualizations that help answer the guiding questions:
- Time-series or trend plots if timing data is available
- Heatmaps to show patterns across dimensions
- Scatter plots to show data distribution and variability
- Cumulative time-series to display how work or issues have accumulated or distributions have shifted over time
- Prioritise plots that show all data (e.g. scatter) over those that summarise (bar, pie)
- Again, examine carefully the plot examples in the MatplotLib and Seaborn galleries and consider whether any of these visualisations (or variations/combinations thereof) might be useful in this context:
  - https://github.com/matplotlib/matplotlib/tree/main/galleries
  - https://github.com/mwaskom/seaborn/tree/master/examples

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
