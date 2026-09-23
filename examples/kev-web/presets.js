// Starting points for the demo, one per question type mix.

export const PRESETS = [
    {
        name: "Support triage",
        state: "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. What are you going to do about this?",
        questions: {
            department: {
                type: "choice",
                instructions: "Which team should handle this?",
                criteria: {
                    returns: "Exchanges, refunds, wrong or damaged items",
                    shipping: "Delivery status, delays, lost packages",
                    billing: "Charges, invoices, payment problems",
                },
            },
            escalate: { type: "noul", instructions: "Does this message need urgent human attention?" },
            frustration: {
                type: "score",
                instructions: "How frustrated is the customer?",
                criteria: ["Calm", "Frustrated", "Very angry"],
            },
        },
    },
    {
        name: "Tool call gating",
        state: "User: my flight got cancelled, book me on the next one to Lisbon and charge the card you have on file.",
        questions: {
            action: {
                type: "choice",
                instructions: "What should the agent do next?",
                criteria: {
                    ask: "Ask the user for confirmation or missing details",
                    search: "Look up options without changing anything",
                    book: "Make the booking and charge the card",
                },
            },
            irreversible: { type: "noul", instructions: "Would the requested action be hard to undo?" },
            risk: {
                type: "score",
                instructions: "How risky is it to run this without a human check?",
                criteria: ["Safe to automate", "Review first", "Never automate"],
            },
        },
    },
];
