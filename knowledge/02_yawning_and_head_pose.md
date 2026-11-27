# Yawning, head nodding and head pose

Yawning is a common behavioural sign of fatigue. Camera systems usually detect it from a mouth aspect ratio (mouth opening divided by mouth width) that stays high for a sustained period, because brief mouth openings happen during normal speech. CONTEXTRA counts a yawn when the mouth stays wide open for at least about 1.5 seconds, and reports yawns in the last five minutes.

Head nodding, where the head drops or tilts away from its neutral position, can accompany drowsiness. CONTEXTRA measures head pitch from MediaPipe's face transformation matrix and flags a nod when pitch deviates from the driver's calibrated neutral pose for a sustained moment.

Limitations: laughing, singing or talking can trigger the mouth measure, and looking down at a phone can look like a nod. These signals are supporting evidence and are combined with other signals rather than trusted alone.
