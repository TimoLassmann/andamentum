# GROVE weekly sync — 2026-04-30

## Attendees

Sarah, Tegan, Marcus, me. Apologies from Daniel.

## Agenda

1. MAP-Elites archive growth this week
2. Antibody-affinity benchmark — first results
3. Open questions for the wet-lab partners

## MAP-Elites archive growth

Sarah walked through the current archive state. We are at 1842 elites in the 32×32 grid as of this morning, up from 1304 last Friday. Coverage is now 18% of the grid; most of the new fills are in the high-affinity / low-developability quadrant which had been almost empty until the new selection operator landed on Tuesday.

Decision: we keep the new selection operator on for at least another week before re-evaluating. Sarah will pull the lineage stats by Friday so we can see whether the new fills are dominated by mutations from a single ancestor or whether we have genuine diversity. Marcus flagged that we should also look at whether the developability scores for the new elites are calibrated against the existing held-out set — if the new operator is producing affinity-high but developability-uncalibrated candidates, we'll want to know before we start downstream.

Action item (me): pull the held-out calibration plot for the new fills and post by Wednesday.

## Antibody-affinity benchmark — first results

Tegan presented the first end-to-end results from the affinity benchmark. The new pipeline correctly recovers the published affinities for 7 out of 9 reference complexes within 0.5 log units, which is in line with the literature for this class of model. The two outliers are both cases where the published binding mode is contested in subsequent papers, so this may not be a real failure mode of our pipeline.

Discussion turned to whether we should publish a benchmark note before the full GROVE paper. Marcus argued for publishing the benchmark separately so we can establish methodological credibility ahead of the bigger claims. Sarah was concerned about timing — if we publish the benchmark first and then GROVE in close succession, the GROVE paper will be reviewed against our own benchmark, which feels circular.

Decision: defer the publish-or-not call until we see results on a second, independent benchmark. Tegan will scope the second benchmark this week.

Action item (Tegan): identify a second affinity benchmark by Monday and circulate.

## Open questions for the wet-lab partners

Daniel had emailed three open questions ahead of the meeting:

1. Are the wet-lab partners willing to test the top 5 candidates from the latest MAP-Elites run, or do they want to wait until we have a more curated short-list?
2. What's the realistic cycle time for binding measurements right now, given the equipment maintenance schedule?
3. Should we be filtering candidates by manufacturability before we send them, or is that something they'd rather assess themselves?

We agreed I will draft a short note to Daniel summarising our current thinking and asking him to ping the wet-lab team. Action item (me): draft the note by Friday.

## Next meeting

Same time next week. Sarah will chair if I'm at the workshop.
