# Questions for the adviser

1. **Venue and timeline.** PeerJ Computer Science (journal; the closest prior work is there),
   the MICCAI UNSURE workshop (thematic fit, short paper), or an IEEE Region 10 conference?
   Each sets a different length, deadline and APC. Is APC funding or a waiver available
   through PUP?
2. **Third-party code.** The 2025 notebook and GUI were partly adapted from an unlicensed
   student project (omertortumlu/midterm-project). They have been removed from the default
   branch and credited. Should the team also tell the original authors, and should the paper
   acknowledge them?
3. **Ethics.** The study uses only public, de-identified CC BY 4.0 data. Does the PUP
   research ethics office require an exemption letter?
4. **Local motivation.** Is there a citable Philippine source on access to andrology or
   fertility testing (DOH, PhilHealth, a local study)? If not, the Philippine framing will
   be cut to one sentence without statistics.
5. **Clinical input.** Could the team reach an embryologist or medical technologist, even
   informally? Two tasks would strengthen the paper: re-labelling the 4 SMIDS label-conflict
   pairs, and giving qualitative feedback on the refer-to-expert design. A small System
   Usability Scale study would need ethics clearance.
6. **Compute.** Does the college have a GPU (or credits) the team could use for one
   additional fine-tuned transformer (e.g. BEiT-Base)? It would answer the obvious reviewer
   question about stronger models. The pipeline needs only a new YAML file.
7. **Authorship and AI disclosure.** Please confirm the author order, CRediT roles, and how
   to disclose AI assistance under the chosen venue's policy.
8. **Statistics review.** The test choices (corrected resampled t-test, per-image Wilcoxon,
   Holm, bootstrap CIs) are documented in Chapter 3 §3.9. Would the statistics instructor
   review them?
