# Harbor & Vale Flow

A CrewAI Flow that connects two crews with a machine-enforced data contract.

- **Crew 1 (Data Analyst):** ingests and cleans retail sales data, then publishes `dataset_contract.json`.
- **Crew 2 (Data Scientist):** reads only the contract, validates the data against it, and trains regression models (target: `Total Amount`).
- **Flow:** runs both crews, validates in plain Python between them, and stops with a clear message if validation fails.

Why: a silent cents-to-dollars change once broke a model for five weeks. The contract makes that kind of change fail loudly.

Setup: `python -m venv .venv`, `.venv\Scripts\activate`, `pip install -r requirements.txt`, copy `.env.example` to `.env`.
