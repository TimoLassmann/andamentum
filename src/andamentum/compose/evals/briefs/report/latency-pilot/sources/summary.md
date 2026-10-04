# Latency pilot summary

The Apollo Labs latency pilot compared the old request router with a new router over eight weeks.
Both routers handled live traffic of about 2 million requests per week.
The new router lowered median latency by 12 percent over the eight weeks.
The 99th percentile latency dropped from 410 milliseconds to 330 milliseconds.
The new router raised memory use by 9 percent on each server.
Two incidents occurred in week 5, both caused by a misconfigured retry limit, and both were fixed within one hour.
The engineering board asked for a recommendation on whether to roll the new router out to all regions.
