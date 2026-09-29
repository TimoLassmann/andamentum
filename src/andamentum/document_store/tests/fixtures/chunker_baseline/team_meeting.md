# SaltLog weekly sync — 2026-04-30

## Attendees

Priya, Jonas, Aiko, me. Apologies from Felix.

## Agenda

1. Sensor uptime this week
2. Calibration drift — first comparison
3. Open questions for the harbour authority

## Sensor uptime

Priya walked through the dashboard. We have 38 of 42 nodes reporting as of this morning, up from 31 last Friday. The four silent nodes are all on the northern jetty, where the new mounting brackets went in on Tuesday, so the first suspect is the antenna placement rather than the batteries.

Decision: we leave the northern nodes as they are for one more week before we swap the brackets back. Priya will pull the signal-strength logs by Friday so we can see whether the dropouts line up with the tide or with the time of day. Jonas pointed out that the two oldest nodes are still on the first firmware — if the dropouts cluster there, the bracket theory is wrong and we need to know before the next site visit.

Action item (me): plot the uptime per node against the tide table and post by Wednesday.

## Calibration drift

Aiko presented the first side-by-side of our sensors against the reference probe at the boat ramp. Over nine days the median offset grew from 0.2 to 0.9 practical salinity units, and the growth is roughly linear, which points at biofouling on the electrodes rather than a fault in the readout.

Discussion turned to whether we should publish the drift curve now or wait for a second site. Jonas argued for waiting: one site cannot separate fouling from local water chemistry. Aiko argued that the curve is useful to anyone building a similar network and that a short note costs little.

Decision: defer the publish-or-not call to the next sync, once a second site has two weeks of data.

Action item (Aiko): pick a second site with a reference probe and start logging.

## Harbour authority questions

Felix had emailed three open questions from the harbour authority. First, can we report hourly means instead of raw readings. Second, what happens to the data if a node is lost in a storm. Third, whether the network could also log water temperature at no extra cost.

We agreed I will draft a short reply to Felix covering the first two and flag the third as a hardware question for Jonas.

## Next meeting

Same time next week. Priya to chair.
