import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class PhishingDetectionAgentWithGraph:
    def __init__(self):
        """Initialize with a trained ML model."""
        self.model = None
        self._train_ml_model()

    def _train_ml_model(self):
        """Train a simple ML model using synthetic data."""
        np.random.seed(42)
        data = np.random.randint(0, 2, (100, 9))
        labels = np.random.randint(0, 2, 100)

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"ML Model trained with accuracy: {accuracy:.2f}")

    def extract_features(self, email_content):
        """Extract phishing-related features from email content."""
        features = {
            'has_urgent_words': int(bool(re.search(r'urgent|immediate|alert', email_content.lower()))),
            'has_financial_words': int(bool(re.search(r'account|credit|payment', email_content.lower()))),
            'has_security_words': int(bool(re.search(r'password|login|verify', email_content.lower()))),
            'has_threat_words': int(bool(re.search(r'suspended|expired|terminate', email_content.lower()))),
            'suspicious_url_count': len(re.findall(r'https?://[^\s]+', email_content)),
            'ip_address_url': int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', email_content))),
            'mismatch_from_domain': np.random.randint(0, 2),  # Placeholder
            'reply_to_mismatch': np.random.randint(0, 2),  # Placeholder
            'subject_has_urgent': np.random.randint(0, 2)  # Placeholder
        }
        return features

    def predict_phishing_probability(self, features):
        """Predict phishing probability using ML model."""
        feature_vector = np.array([list(features.values())])
        probability = self.model.predict_proba(feature_vector)[0][1]  # Probability of phishing
        return probability

    def analyze_email(self, email_content):
        """Analyze email and return combined phishing risk."""
        features = self.extract_features(email_content)
        ml_probability = self.predict_phishing_probability(features)

        # Combine ML probability with rule-based heuristic
        rule_based_risk = sum(features.values()) / len(features)
        combined_risk_score = (ml_probability * 0.6) + (rule_based_risk * 0.4)

        if combined_risk_score >= 0.7:
            risk_level = "High"
        elif combined_risk_score >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "ml_probability": ml_probability,
            "rule_based_risk": rule_based_risk,
            "combined_risk_score": combined_risk_score,
            "risk_level": risk_level,
            "features": features
        }

    def visualize_risk(self, email_analyses):
        """Generates a bar chart comparing ML probability vs Rule-Based Risk."""
        emails = [f"Email {i + 1}" for i in range(len(email_analyses))]
        ml_probabilities = [analysis["ml_probability"] * 100 for analysis in email_analyses]  # Convert to percentage
        rule_based_risks = [analysis["rule_based_risk"] * 100 for analysis in email_analyses]

        plt.figure(figsize=(10, 5))
        bar_width = 0.4
        index = range(len(emails))

        plt.bar(index, ml_probabilities, bar_width, label="ML Predicted Probability", color='blue', alpha=0.7)
        plt.bar([i + bar_width for i in index], rule_based_risks, bar_width, label="Rule-Based Risk Score", color='red',
                alpha=0.7)

        plt.xlabel("Emails Analyzed")
        plt.ylabel("Risk Score (%)")
        plt.title("Comparison of ML Probability vs Rule-Based Risk")
        plt.xticks([i + bar_width / 2 for i in index], emails, rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save the figure as a screenshot
        plt.savefig("phishing_risk_analysis.png")
        plt.show()


# === RUN THE SCRIPT ===
if __name__ == "__main__":
    agent_with_graph = PhishingDetectionAgentWithGraph()

    # Sample phishing emails for testing
    sample_emails = [
        "Urgent! Verify your PayPal account now: http://paypal-secure.tk/login",
        "Limited-time offer! Claim your free gift: http://gift-card.xyz",
        "Your Netflix subscription is expiring. Update your payment details now.",
        "Security Alert! Someone tried to access your bank account.",
        "Exclusive deal for you: Buy now and get 50% off!"
    ]

    sample_analyses = [agent_with_graph.analyze_email(email) for email in sample_emails]

    # Generate graph visualization
    agent_with_graph.visualize_risk(sample_analyses)

