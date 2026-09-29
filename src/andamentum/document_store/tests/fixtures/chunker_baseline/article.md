# SaltLog: A Low-Cost Sensor Network for Estuary Salinity Monitoring

## Abstract

Estuary salinity is measured today in one of two ways, and both leave gaps. Fixed reference stations are accurate but expensive, so an estuary of any size carries only a handful of them, and the spatial picture between stations is interpolated rather than observed. Ship-based surveys fill in the spatial picture but only for the few days a year a vessel is available, so short events such as a storm surge or a flushing pulse after heavy rain are almost never captured. We present SaltLog, a network of low-cost conductivity sensors designed to be deployed in dozens rather than in ones, each node costing less than a tenth of a reference station.

We describe the node hardware, a two-point field calibration procedure that can be run by one person from a small boat, and a data pipeline that flags drift and fouling automatically. Over a six-week deployment of 42 nodes in a temperate estuary, the network reported 91% of expected readings, tracked a reference probe to within 0.5 practical salinity units after drift correction, and resolved a saltwater intrusion event that the two existing reference stations placed several hours apart.

## Introduction

Salinity is one of the most basic quantities an estuary manager needs. It sets which species can live where, it controls how sediment settles, and it is the first signal of a change in freshwater inflow, whether from a dry season, an upstream dam release, or a flood. Because it changes with the tide, with river flow and with wind, a salinity field is never static, and the interesting structure is often at scales of hundreds of metres and a few hours.

The standard instrument is a conductivity, temperature and depth probe, either mounted at a fixed station or lowered from a vessel. A fixed station reports continuously and accurately, but the cost of the probe, the mounting, the power supply and the telemetry means that most estuaries have between one and five of them. Everything between the stations is inferred. Vessel surveys give a dense spatial snapshot, but a survey is a single moment, and the moments that matter most — a storm, a king tide, the first flush after a drought — are exactly the ones a vessel is least likely to be out in.

Low-cost environmental sensing has changed what is practical in air quality and in soil moisture, where networks of hundreds of cheap nodes now sit alongside a few reference instruments. The cheap nodes are individually less accurate, but their density lets an analyst see structure that no sparse network could show, and the reference instruments anchor their calibration. Water is a harder environment than air: electrodes foul, housings leak, and radio signals do not travel through water, so a node has to surface or sit above the waterline to report.

Several groups have built low-cost conductivity sensors for teaching or for single-site studies, and a few have reported small networks of five to ten nodes. What has been missing is a design that scales to dozens of nodes without a matching increase in maintenance effort. In our experience the limiting cost is not the parts but the time: every node that needs a site visit for recalibration or cleaning adds a boat trip, and a network that needs weekly visits to every node is not a low-cost network at all.

In this paper we present SaltLog, a network designed around that constraint. Each node is built from off-the-shelf parts, mounts on existing jetty piles and channel markers, and reports over a long-range radio link to a single shore gateway. Calibration is done in the field with two reference solutions, and drift between visits is estimated from the data itself by comparison with the nearest reference station. We describe the design, report a six-week deployment, and discuss where the approach works and where it does not.

## Related work

Low-cost water quality sensing has a long history in citizen science and education. Early designs used a pair of stainless steel electrodes and a simple voltage divider, which is enough to tell fresh from salt water but drifts badly and responds to temperature as much as to salinity. Later designs added temperature compensation and alternating-current excitation, which removes most of the electrode polarisation that made the early readings unstable.

Networked designs are more recent. Small deployments of five to ten nodes have been reported in lakes, in aquaculture ponds and in one estuary, usually with each node logging to a memory card that is collected on a site visit. Real-time networks with radio telemetry are rarer, and most report on the radio link rather than on the measurements, because keeping a link alive over water is the hard part of the engineering.

Drift and fouling are the two problems every long deployment reports. Biofouling builds a film on the electrodes that changes the cell constant, and the change is gradual enough to look like a real trend in the water. The usual response is frequent cleaning, which is exactly the maintenance cost a large network cannot afford. Statistical approaches to drift correction exist for air quality sensors, where a cheap sensor is compared continuously with a co-located reference, but they have seen little use in water.

Our contribution sits between these threads. We use a conventional four-electrode cell with temperature compensation, a long-range radio link, and a drift model borrowed from the air quality literature, and we evaluate the combination at a scale larger than any previous estuary deployment we are aware of.

## Methods

The network has three parts: the sensor nodes, a shore gateway that receives their reports, and a server that stores the readings and runs the quality control. We describe each in turn, followed by the calibration and deployment procedures.

### Sensor hardware

Each node is built around a low-power microcontroller, a four-electrode conductivity cell, a digital temperature sensor, and a long-range radio module. The conductivity cell uses alternating-current excitation at 1 kHz to avoid polarisation, and the four-electrode layout makes the reading largely insensitive to the resistance of the outer electrodes, which is where fouling builds up first.

The electronics sit in a sealed housing above the waterline, and the cell hangs below it on a short cable inside a perforated guard tube. The guard keeps floating debris off the electrodes and gives a fixed geometry, so the cell constant does not change when the node is knocked by a boat or a log. Power comes from a small solar panel and a lithium battery sized for ten days without sun.

A node wakes every ten minutes, takes five conductivity readings and one temperature reading, reports the median conductivity and the temperature, and goes back to sleep. The median is robust to a single bad reading caused by a bubble passing through the cell. At this duty cycle the average current draw is low enough that a node runs indefinitely on the solar panel in all but the darkest winter weeks.

### Calibration

Each node is calibrated in the field against two reference solutions that bracket the expected salinity range. The operator lowers the cell into each solution in turn from a small boat, waits for the reading to settle, and records the result on a handheld device that sends the calibration pair to the server. The whole procedure takes about four minutes per node.

The two-point calibration gives a gain and an offset for each node. We found in bench tests that a third point added little: the cell response is close to linear across the range we care about, and the dominant error in the field is drift after calibration rather than non-linearity at calibration time.

Calibration is repeated at every site visit, and the difference between the old and new calibration is recorded as a direct measurement of the drift since the last visit. Those measurements are what we use to validate the drift model described below.

### Deployment

The deployment ran for six weeks in a temperate estuary with a tidal range of about two metres and a single main river inflow. We placed 42 nodes along the main channel and in three side channels, spaced so that each node was within 800 metres of at least two others. Nodes were mounted on existing jetty piles, channel markers and mooring posts, with permission from the harbour authority, so no new structures were needed.

Two existing reference stations operated by the harbour authority sat within the deployment area, one near the mouth and one about six kilometres upstream. We used both as anchors for the drift model and one of them as the validation reference, holding it out of the model when reporting accuracy.

Site visits were scheduled every two weeks. At each visit the operator cleaned the guard tube, recalibrated the node, and checked the housing for leaks. Three nodes were lost during the deployment: one to a boat strike, one to a failed housing seal, and one that was removed by a member of the public and later returned.

The shore gateway sat on a building near the middle of the estuary, with a clear line of sight along the main channel. Nodes in the side channels relayed through the nearest main-channel node when they could not reach the gateway directly.

### Data pipeline

Readings arrive at the gateway as short radio packets carrying the node identifier, a sequence number, the median conductivity and the temperature. The gateway forwards each packet to the server, which converts conductivity to practical salinity using the node's current calibration and the temperature reading, and stores both the raw and the converted values.

Storing the raw value matters because the calibration changes at every site visit and the drift model revises its estimates as new reference data arrive. Any salinity value can therefore be recomputed from the raw record, and we never overwrite a stored raw reading.

The sequence number lets the server tell a lost packet from a node that did not wake. A gap in the sequence with the node reporting on either side is a radio problem; a gap in the sequence that ends with a battery warning is a power problem. The distinction decides whether a site visit is needed.

Drift correction runs once a day. For each node, the model compares the node's readings during slack water with the reading of the nearest reference station at the same time, fits a linear drift term since the last calibration, and subtracts it. Slack water is used because the salinity field is most uniform then, so the difference between a node and the reference is mostly drift rather than real spatial structure.

### Quality control

Every reading passes three automatic checks before it is shown on the dashboard. A range check rejects values outside the physically possible range for the estuary. A rate check rejects changes faster than the tide can produce between two consecutive readings. A neighbour check flags a node whose reading departs from both of its neighbours by more than a set margin for more than an hour.

Flagged readings are kept but marked, and the dashboard shows them in a different colour. A node that stays flagged for a full day is added to the list for the next site visit.

### Implementation and release

The node firmware, the gateway software and the server pipeline are open source. The server is a small Python application with a time-series database behind it, and it runs on a single virtual machine. The hardware design, including the housing drawings and the parts list, is published alongside the code so that another group can build the same nodes.

## Results

Over the six weeks the network delivered 91% of the readings it would have delivered with perfect uptime. Most of the missing readings came from four nodes on the northern jetty during the first two weeks, where the mounting placed the antenna below the level of the jetty deck; moving the antenna 30 centimetres higher at the first site visit brought all four back.

After drift correction, the node closest to the held-out reference station tracked it with a median absolute difference of 0.4 practical salinity units and a 95th percentile of 1.1. Without drift correction the median difference grew steadily after each calibration and reached 1.3 units by the end of a two-week interval, which matches the drift we measured directly at site visits.

The drift model's estimate of each node's drift at the moment of recalibration can be compared with the drift measured by the recalibration itself. Across 118 site-visit recalibrations the two agreed to within 0.3 units in 84% of cases. The largest disagreements came from nodes in a side channel with weak tidal exchange, where slack water is less uniform and the comparison with the reference station is less clean.

The neighbour check flagged 2.1% of readings. On inspection most flags came from real short-lived structure — a plume of fresher water from a stormwater outlet passing one node before it mixed — rather than from sensor faults. We discuss this trade-off below.

## Intrusion event — week three

In the third week a period of low river flow coincided with a spring tide, and a wedge of saltier water moved further upstream than at any other time during the deployment. The two reference stations recorded the arrival of the saltier water about four hours apart, and on their data alone the event looks like a single front moving steadily up the channel.

The network shows a different picture. The saltier water moved up the main channel in two pulses on consecutive flood tides, and between the pulses it partly retreated. The first pulse stalled at a bend about three kilometres upstream, where the channel narrows, and only the second pulse passed the bend. Two of the side channels filled from the main channel during the second pulse and stayed saltier for more than a day after the main channel had freshened again.

None of this structure is visible in the reference record. The reference stations sit either side of the bend, so they see the start and the end of the intrusion but not the stall between. The side-channel retention is invisible to them altogether, because neither station is in a side channel.

We checked the network's account against the one independent measurement available: a vessel survey that happened to run on the morning between the two pulses. The survey found the saltier water at the bend and not beyond it, which agrees with the network and not with a single steady front.

The event also tested the pipeline. Salinity at several nodes changed faster than at any other time in the deployment, and the rate check was close to rejecting real readings. We widened the rate limit for the side channels after the event, and we report the widened limit in the released configuration.

## Discussion

SaltLog shows that a dense network of low-cost nodes can resolve estuary salinity structure that a sparse network of reference stations misses, and that the maintenance cost of such a network can be held at one visit per node every two weeks. The main enabling step is not the hardware, which follows established designs, but the combination of a drift model with the reference stations that already exist.

The drift model depends on those reference stations. A node far from any reference, or in a channel whose slack-water salinity differs systematically from the reference, is corrected less well, and our side-channel results show the cost. An estuary with no reference station at all could not use the model as described; it would need at least one well-maintained reference node of its own, which raises the cost of the network but not by much.

The neighbour check illustrates a general tension in quality control for dense networks. The check is designed to catch a faulty node, and it does, but it also catches a real feature that passes one node before its neighbours see it. A denser network makes real small-scale structure more common, not less, so a check tuned on a sparse network will flag more real events as the network grows. We chose to keep flagged readings visible rather than discard them, and we think that is the right default for any network whose purpose is to see structure the reference stations cannot.

Fouling set the site-visit interval. We tried a copper mesh on the guard tube at five nodes and saw slower drift at those nodes, but five nodes over six weeks is too few to draw a conclusion, and copper in an estuary raises its own questions. A longer deployment with a proper comparison is the obvious next step for anyone adopting the design.

A limitation of the deployment is its length. Six weeks covers one intrusion event and no major flood, so we cannot say how the network behaves when the river is in flood and debris loads are high. The three lost nodes suggest that losses would rise in those conditions, and a network planning for flood monitoring should budget for them.

## Conclusion

We have presented SaltLog, a network of low-cost salinity sensors that can be built, calibrated and maintained at a scale of dozens of nodes. Over a six-week deployment it delivered 91% of expected readings, tracked a held-out reference to within half a salinity unit after drift correction, and revealed the two-pulse structure of an intrusion event that the existing reference stations recorded as a single front. The hardware, firmware and pipeline are released so that other groups can reproduce the network and test it in other estuaries.
