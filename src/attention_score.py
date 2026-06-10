class AttentionScorer:

    def __init__(
        self,
        ear_threshold=0.20,
        yaw_threshold=15,
        pitch_threshold=15
    ):

        self.ear_threshold = ear_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

    def classify(self, ear, yaw, pitch):

        score = 0

        # Eye Status
        if ear >= self.ear_threshold:
            score += 1

        # Looking Left/Right?
        if abs(yaw) <= self.yaw_threshold:
            score += 1

        # Looking Up/Down?
        if abs(pitch) <= self.pitch_threshold:
            score += 1

        if score >= 2:
            return "Attentive", score

        return "Distracted", score