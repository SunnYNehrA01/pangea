# PANGEA – City Pollution Pattern Analysis and Insights

Air pollution is a global health crisis. It causes an estimated 7 million premature deaths each year and 99% of people breathe dangerously polluted air[1][2]. Urban air is especially harmful to vulnerable communities (often the poor) and also contributes to climate change[3][2]. To tackle this, city governments worldwide are enacting measures (e.g. electrifying transit, banning dirty fuels, public education campaigns) that have begun to improve air quality in places like Bogotá, Warsaw, Seoul and Accra[4][5]. Long-term trend analysis of air-quality data can reveal whether such interventions are working and guide new policies[6].

WHO data show that nearly all countries are above recommended limits and many have no formal standards at all[2][7]. For example, WHO notes that “almost all of the global population (99%) are exposed to air pollution levels that put them at increased risk”[2]. The map (right) highlights this disparity: many nations lack strict particulate limits (shown in brown) or exceed them widely. In this context, PANGEA was born to give citizens and decision-makers a powerful, data-driven tool to understand and act on their city’s pollution story, by learning from history rather than guessing at solutions.

---

## Goals and Objectives

- **Analyze city air pollution data.**  
  PANGEA takes a city’s historical pollution measurements (PM₂.₅, NO₂, O₃, etc.) and meteorological factors, and applies exploratory data analysis and statistical models to uncover patterns – e.g. seasonal spikes in particulates, correlations with traffic or industry, or the impact of local events.

- **Match with historical cases.**  
  The system then compares the target city’s pattern to a library of 170+ case studies from other cities and time periods. Each case records real pollution events and policies (e.g. smog episodes, new regulations or fuel changes). By finding similar data signatures, PANGEA identifies which historic precedents most closely resemble the current situation. As one WHO report emphasizes, lessons from diverse city case studies (New York, Istanbul, Toronto, Beijing, etc.) provide valuable guidance for other cities facing pollution challenges[8][9].

- **Provide actionable recommendations.**  
  For each matched case, the platform summarizes what policies or changes were implemented and how effective they were. These recommendations are assembled into a context summary and fed to a GPT-2 language model (via Hugging Face) to generate clear, city-specific advice. This AI-powered insight helps translate technical analysis into practical steps for both the public and authorities.

- **Empower public awareness and accountability.**  
  By exposing the data and comparisons, PANGEA enables ordinary citizens to understand why pollution is happening and what can be done. The tool also assigns each city a Pollution Management Score (considering pollution trends, response measures, and outcomes). A low score signals that more action is needed. The goal is to make it easy for residents and media to question officials if air quality is worsening and to learn from other cities’ successes.

---

## Key Innovations

- **Historical pattern matching:**  
  Unlike conventional dashboards that only show current data, PANGEA actively searches for analogous pollution histories. This case-based approach is rare in environmental apps. For example, one study used time-series clustering and neural nets to recognize that traffic and industry drives pollution patterns[10]. PANGEA automates such reasoning: by linking data to documented past events, it helps avoid repeating mistakes and leverage proven solutions.

- **AI-driven interpretation:**  
  The platform not only crunches numbers but also uses a language model (GPT-2) to generate narrative insights. After identifying the top 3 historical matches, it composes a contextual prompt summarizing the target city’s data and the matched policies, then queries GPT-2. This yields written recommendations and explanations that are easier for non-experts to understand.

- **Composite scoring:**  
  Each city-year combination gets a Performance Score reflecting its pollution handling. Rather than just pollution levels, the score factors in things like policy interventions, pollution reduction rates, and comparisons to matched cases. In effect, this score serves as a “report card” on how well the city’s leadership has addressed the issue. Developing such indices is informed by multi-criteria environmental assessments seen in the literature[6], but PANGEA’s score is uniquely tailored to each city’s context.

---

## Implementation and Phases

1. **Data Collection:**  
   Integrated multiple data sources, including government air monitoring stations and satellite records (e.g. open datasets on PM₂.₅, NO₂, O₃, weather, traffic indices). Data from fixed sensors was complemented with modeled estimates to fill gaps[11][12].

2. **Data Cleaning & EDA:**  
   Used Python (pandas, NumPy) to clean time series (handling missing values, smoothing outliers) and performed exploratory analysis. Identified seasonal cycles (e.g. winter PM₂.₅ spikes, summer ozone peaks) and visualized hotspots[13].

3. **Historical Case Library:**  
   Compiled 170+ documented episodes (news archives, WHO/UNEP reports, published case studies) spanning major pollution events and policy shifts. Each case includes pollutant timelines and annotations.

4. **Pattern Matching Engine:**  
   Implemented statistical techniques (clustering, PCA) to quantify similarity between current city data and case studies[10].

5. **Scoring Algorithm:**  
   Defined rubric factoring AQI trends and policies to rate city pollution management[14].

6. **AI Insight Generation:**  
   Top 3 matching case contexts combined with current city data fed to GPT-2 model to generate actionable insights.

7. **Web Interface:**  
   Built with Flask backend and HTML/CSS/JS frontend; allows multi-city, time-range selection and displays trend charts, scores, and AI recommendations.

---

## Features

- Multi-city selection: Compare up to 5 cities side-by-side for a chosen year/month.
- Time-range flexibility: Data from 2018 through 2025.
- Pollution trend visualizations: Interactive line graphs and maps of PM₂.₅, NO₂, O₃, etc.
- Case-match highlights: Show top 3 historical cases with similarity scores and policy summaries.
- Performance scoring: Pollution Management Score indicating efficacy of pollution control.
- AI Recommendations: Tailored, AI-written insights grounded in matched cases.
- User guidance: Tooltips and legends help non-experts understand technical terms.

---

## Societal Impact

PANGEA bridges data with public empowerment. By making analysis transparent, ordinary people can learn why their city’s air is polluted and what can be done. For example, in Accra, public education on cookstoves and waste burning was key to improving air quality[5]; similarly, PANGEA educates users on local causes. It also promotes accountability by enabling citizens to question officials if scores fall or policies lack.

For policymakers and planners, PANGEA offers evidence-based decision support. Historical trend analysis helps evaluate policy success, as seen in cities like New York and Istanbul that reduced smog without harming economic growth[15][9]. PANGEA aligns health and climate goals by showing how cleaner air reduces greenhouse emissions[2][3].

---

## Future Work

- Global scaling to include all major cities worldwide to improve case matching.
- Context-aware AI agents factoring in local geography, climate, and demographics.
- Integration of real-time sensor data including crowdsourced and IoT monitors.
- Policy impact simulation to predict outcomes of proposed actions (what-if scenarios).

---

## Sources

- Major findings and approaches backed by environmental research and reports (WHO, UNEP, etc.)[2][16][8][15].
- Trend analysis and multivariate statistical techniques validate PANGEA’s methodology[6][10].
- Case studies from cities like New York, Istanbul, Beijing facilitate learning and policy design.

---

## References

1. [Five cities tackling air pollution – UNEP](https://www.unep.org/news-and-stories/story/five-cities-tackling-air-pollution)  
2. [Air pollution data – WHO](https://www.who.int/data/gho/data/themes/air-pollution)  
3. Climate change and air pollution reports  
4. & 5. City case studies on air quality improvements  
6. Historical air quality trend analyses  
8., 9., 15. Environmental Governance & Air Quality reports  
10. [Spatial and temporal air quality pattern recognition, Malaysia study](https://pubs.rsc.org/en/content/articlelanding/2013/em/c3em00161j)  
11., 12., 13., 14. Data modeling and scoring literature  

---

*End of README*
