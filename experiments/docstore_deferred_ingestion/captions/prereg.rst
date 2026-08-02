**The pre-registration.** Every hypothesis, its numeric threshold, its
falsifier and the scoring rules — written **before** any measurement exists
(this rule's only dependency is ``provenance.json``, so that the pre-registration
carries the run's real ``run_id``; no measurement has happened at that point).

``analyze.py`` compares against this file and nothing else. Registering the
*scoring rules* here too closes the obvious loophole: a threshold that cannot
be changed after the fact is worth little if the rule for turning a measurement
into a verdict can be.
