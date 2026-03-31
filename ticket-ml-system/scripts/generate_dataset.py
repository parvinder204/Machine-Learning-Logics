import random
import pandas as pd
from pathlib import Path

SEED = 42
random.seed(SEED)

TEMPLATES = {
    "billing": [
        "My payment failed but the amount was deducted from my account",
        "I was charged twice for the same subscription",
        "I cannot see my invoice for last month",
        "My credit card was charged but the subscription is not active",
        "Why is my bill higher than expected this month?",
        "I need a copy of my payment receipt",
        "Payment declined even though my card is valid",
        "I upgraded my plan but was still charged the old price",
        "Subscription renewed without my consent",
        "I see an unauthorized charge on my statement",
        "How do I update my billing information?",
        "My discount code was not applied to the invoice",
        "I need to change my payment method",
        "Can I get an itemized bill for my account?",
        "Transaction failed but money was deducted",
    ],
    "technical_issue": [
        "The app keeps crashing when I try to open it",
        "I cannot upload files, the page freezes",
        "Error 500 appears every time I click submit",
        "The dashboard is not loading correctly",
        "Videos are not playing, just showing a black screen",
        "My data is not syncing across devices",
        "I get a timeout error after 30 seconds",
        "The mobile app is very slow and unresponsive",
        "Notifications are not working on my phone",
        "I cannot export my data to CSV",
        "The search feature returns no results",
        "Images are not displaying in my reports",
        "The login button is grayed out and unclickable",
        "My account settings are not saving",
        "API returns 403 even with correct credentials",
    ],
    "account_access": [
        "My account is locked and I cannot log in",
        "I forgot my password and the reset email is not arriving",
        "Two-factor authentication is not working for me",
        "I cannot access my account after changing my email",
        "My account was suspended without any notification",
        "I need to merge two accounts into one",
        "How do I transfer my account to another email?",
        "I am locked out after too many login attempts",
        "My SSO login is redirecting to an error page",
        "I cannot change my username in account settings",
        "My session expires too quickly, I keep getting logged out",
        "I need to add a team member to my account",
        "How do I delete my account permanently?",
        "I cannot verify my phone number for 2FA",
        "My admin access was revoked but I need it restored",
    ],
    "refund": [
        "I want a refund for the subscription I just purchased",
        "Refund not processed after 14 days",
        "I was promised a refund but have not received it yet",
        "I accidentally bought the wrong plan, please refund",
        "The product did not work as advertised, I want my money back",
        "I canceled within the trial period but was still charged",
        "How long does a refund take to appear on my statement?",
        "I need a refund for a duplicate payment",
        "My refund was only partial, I expected the full amount",
        "I received a defective product and need a refund",
        "The service was unavailable for days, I deserve compensation",
        "I was billed after canceling, please refund",
        "Refund request submitted a week ago, still no update",
        "Can I get a refund if I did not use the service?",
        "I need to dispute a charge from three months ago",
    ],
    "feature_request": [
        "Please add dark mode to the application",
        "It would be great to have bulk export functionality",
        "Can you add keyboard shortcuts for common actions?",
        "I would love a mobile app for iOS",
        "Please add support for CSV import",
        "Can we get Slack integration?",
        "Would be useful to have role-based access control",
        "Please add an audit log for admin actions",
        "I need the ability to customize email templates",
        "Can you add a calendar view for scheduling?",
        "Please support multi-language localization",
        "It would help to have real-time collaboration features",
        "Can we get webhook support for integrations?",
        "Please add two-factor authentication options",
        "I would like a public API for my developers",
    ],
}

def add_noise(text: str) -> str:
    """Simulate real user typing — typos, truncation, informal phrasing."""
    transforms = [
        lambda t: t.lower(),
        lambda t: t.rstrip("?!.") + "!!",
        lambda t: t.replace("I ", "i "),
        lambda t: "hi, " + t,
        lambda t: t + " please help asap",
        lambda t: t + " this is urgent",
        lambda t: t[:int(len(t) * 0.8)],  # truncate
        lambda t: t,  # no change
    ]
    return random.choice(transforms)(text)


def generate_dataset(n_per_class: int = 300) -> pd.DataFrame:
    rows = []
    for label, templates in TEMPLATES.items():
        for _ in range(n_per_class):
            base = random.choice(templates)
            text = add_noise(base)
            rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(n_per_class=300)
    out_path = out_dir / "tickets.csv"
    df.to_csv(out_path, index=False)

    print(f" Dataset saved → {out_path}")
    print(f" Total rows  : {len(df)}")
    print(f" Distribution:\n{df['label'].value_counts().to_string()}")