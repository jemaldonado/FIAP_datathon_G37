# Datathon 7MLET - POS TECH

## CHALLENGE

The Datathon proposes a unique challenge in the regulated financial domain: design an adaptive experimentation platform for offers, messages, or next steps across digital channels. Each group builds an end-to-end Machine Learning Engineering solution and demonstrates how it would be operated with safety, observability, evaluation, and governance.

The goal is not to replicate a real banking system, but rather to demonstrate technical maturity: formulate the problem, build baselines, version data, serve components, evaluate quality, monitor risk, document limitations, and explain decisions to technical and business audiences, considering the following scenario:

### Use Case

A digital financial institution needs to decide, across different channels, which offer, message, or next step to present to each eligible customer. Fixed rules and long A/B tests waste traffic, are slow to react to context changes, and hinder responsible personalization. This is the core of a **multi-armed bandit** approach: identify distinct behaviors, balance exploration and exploitation, and learn from observed responses without freezing decisions into static rules. The solution must include an LLM-based assistant that summarizes experiments, retrieves synthetic internal policies, and explains decisions.

### Algorithmic References

| Algorithm | Role in Challenge | Expected Evidence |
|-----------|------------------|-------------------|
| **Thompson Sampling** | Bayesian exploration under uncertainty to model conversion, clicks, or expected reward per arm. | Documented priors, comparison with baseline, and exploration analysis. |
| **lil'UCB** | UCB family to select actions based on expected reward and uncertainty. | Formula, implementation, or justified adaptation, with analysis of the trade-off between confidence, exploration, and conversion. |
| **Deterministic Baseline** | Simple control policy (fixed rule, best historical arm, or initial segmentation). | Clear comparative metric showing gain or limitation of the adaptive policy. |

Groups may implement Thompson Sampling, lil'UCB, LinUCB, or another contextual variation, provided they explain the choice, show how context enters the decision, and document the handling of delayed rewards.

## Data, Rules, and Kaggle Datasets

Use a Kaggle dataset compatible with marketing, offers, propensity, campaigns, recommendation, or conversion as a factual reference and create a synthetic layer of adaptive experimentation on top of it (impressions, actions, context, rewards, delayed events, business policy documents, and suitability for RAG). 

**Do not use real customer data, identifiers, wealth, income, gender, race, or private business rules. Keep sensitive decisions in the human loop and document legal basis, purpose, minimization, and retention.**

### Recommended Kaggle Datasets

| Kaggle Dataset | How to Use in Challenge |
|---|---|
| **bank-marketing** (henriqueyamahata) | Banking campaigns, conversion propensity, and offer decision. |
| **bank-marketing-data-set** (tunguz) | Variation of the banking marketing problem for comparison. |
| **bank-term-deposit-subscription** (dharmik34) | Term deposit subscription as a proxy for conversion. |
| **telemarketing-jyb-dataset** (aguado) | Contact campaigns and responses, useful for channel or approach comparison. |

Other datasets are accepted if the group justifies adherence and documents source, version, license, columns, target, and limitations. Discard columns that cause data leakage (e.g., duration in Bank Marketing) and preserve the Kaggle reference.

### Minimum Data Deliverable

Each group must deliver, at a minimum, a versioned derived layer containing:

- `data/kaggle/README.md` (source, link, version, license);
- `data/processed/` (cleaned dataset without data leakage);
- `data/synthetic_enrichment/` with offer_catalog, offer_events, and delayed_rewards;
- `data/golden_set/evaluation_cases.jsonl` with at least 20 cases;
- `reports/data-generation.md` documenting process, seeds, hypotheses, limitations, and risks.

## Mandatory Deliverables

The deliverables below are organized in **nine cumulative stages (0–8)**. Each stage brings an objective, a detailed list of required technical artifacts, and the acceptance evidence criteria that the evaluation panel uses to consider the stage complete.

**A later stage does not compensate for an absent earlier stage**: a group with a sophisticated model but without a traceable dataset, reproducible evaluation, or lifecycle approval is penalized in technical validation. The absence of a sub-item within a stage is not compensated by the presence of others.

### Stage 0 — Project Organization

**Objective**: prepare a public repository that can be reused by another person without oral context from the group.

#### Required Artifacts

- Public repository URL with name following the pattern `datathon-7mlet-group-XX`.
- `README.md` with problem vision, scope, design choices, local execution instructions, folder map, command list, and limitations.
- `pyproject.toml` declaring dependencies, Python version, entry point, and development tools.
- `.env.example` listing required environment variables, without real values.
- License, adequate `.gitignore`, and absence of secrets, sensitive data, or large binary models in version control.
- Commit history showing work evolution, not just a final commit.

#### Acceptance Evidence

An external person can install dependencies, understand the flow, and execute at least one validation command without needing oral explanation from the group.

---

### Stage 1 — Kaggle Dataset and EDA

**Objective**: transform a compatible Kaggle dataset into a trustworthy source for experimentation.

#### Required Artifacts

- `data/kaggle/README.md` with link, version, source, license, limitations, and download instructions.
- Data dictionary, EDA notebook, and quality report.
- Data layer in code that loads the dataset, registers source/version/license, and generates documented derived datasets.
- Documented decision on columns that cause data leakage or post-contact information, with justification for discard or treatment.

#### Acceptance Evidence

The evaluation panel can trace the dataset origin, understand the variables used, and verify that no post-contact information data leakage occurred in the decision model.

---

### Stage 2 — Synthetic Enrichment

**Objective**: create the adaptive experimentation layer on top of the chosen dataset.

#### Required Artifacts

- Synthetic catalog of arms/offers physically separated from the original Kaggle dataset.
- Impression events, decision context, and intermediate rewards with controlled random seeds.
- Delayed rewards modeling and time horizon documentation.
- Schema of synthetic files and generation process described in the repository.

#### Acceptance Evidence

Synthetic files have documented schema and explain how arms, context, reward, and time horizon were defined, separated from the original Kaggle dataset.

---

### Stage 3 — Baseline and Algorithmic Strategy

**Objective**: compare simple policy with multi-armed bandit approach.

#### Required Artifacts

- At least one deterministic baseline implemented.
- Thompson Sampling implementation or simulation.
- Experimental reference or explicit justification of lil'UCB in algorithmic analysis.
- Metrics for reward, regret, exploration, and simulated conversion calculated and reported.
- Cold-start and delayed rewards handling described.

#### Acceptance Evidence

The group presents quantitative comparison between baseline and adaptive policy, with justification of the chosen algorithm and cold-start and delayed rewards handling.

---

### Stage 4 — Offline Evaluation and Golden Set

**Objective**: measure technical quality and risk before serving the decision.

#### Required Artifacts

- Reproducible offline evaluation script or notebook via command line or notebook, with justified metrics.
- Golden set with at least 20 versioned examples in `data/golden_set/evaluation_cases.jsonl` or equivalent format.
- Golden set coverage of typical cases, edge cases, synthetic eligible segments, and adversarial scenarios.
- Each case includes context, expected action, expected reward, justification, and explicit pass/fail criterion.
- Metrics matrix, sensitivity analysis, and fairness analysis of exposure between synthetic segments.

#### Acceptance Evidence

Metrics are reproducible, the golden set is versioned, and the analysis explains limitations, biases, and conditions under which the policy should not be used.

---

### Stage 5 — Demonstrable Service or Interface

**Objective**: expose the decision in a controlled and auditable manner.

#### Required Artifacts

- API, CLI, executable notebook, or demonstrable app that receives context and returns a decision.
- Documented input and output contract, with call example and error handling.
- Auditable decision log with reason codes, selected arm, and policy version applied.
- Single command or script that allows reproducing the end-to-end pipeline in a local environment.
- Minimal test suite covering data contracts, policy, and decision logging.

#### Acceptance Evidence

The evaluation panel can execute a sample decision, see the selected arm, justification, policy version, and generated auditable log.

---

### Stage 6 — Target Azure Architecture

**Objective**: demonstrate how the solution would be operated on Azure.

#### Required Artifacts

- `docs/architecture-azure.md` with Mermaid diagram and Azure services mapping.
- Deployment plan and qualitative cost estimate.
- Coverage of compute, API, data, AI/RAG, observability, security, identity, and governance layers.
- Secrets and credential management plan using Azure Key Vault and Managed Identity.

#### Acceptance Evidence

The architecture uses exclusively Azure, covers the above layers, and justifies trade-offs without relying on another cloud provider.

---

### Stage 7 — MLOps Lifecycle

**Objective**: show how new policies would be tested, approved, and promoted.

#### Required Artifacts

- Retraining plan with promotion criteria, approval gate, rollback, and policy versioning.
- Drift and reward monitoring documented.
- Experiment tracking in MLflow or equivalent tool.
- Test procedure, structured human approval, and controlled production promotion of new policies.

#### Acceptance Evidence

The group demonstrates how a new offer/channel/message hypothesis would move from experiment to controlled production, with human approval and documented rollback.

---

### Stage 8 — Governance, Demo Day, and Reports

**Objective**: close the delivery with operational responsibility and coherent narrative.

#### Required Artifacts

- `docs/model-card.md` with model name, version, training and evaluation data, metrics, intended use, out-of-scope use, fairness analysis, known biases, and technical limitations.

- `docs/system-card.md` with scope, decision flow, dependencies, guardrails, risk scenarios (reward hacking, context manipulation, assistant abuse, suitability violation), and monitoring plan.

- `docs/lgpd-plan.md` with legal basis, purpose, minimization, retention cycle, identifier and protected attribute mapping, log/telemetry policy, and incident response plan.

- **Technical report** of up to 10 pages covering:
  - Problem
  - Chosen dataset
  - Synthetic enrichment
  - Multi-armed bandit modeling
  - Quantitative comparison
  - Target Azure architecture
  - MLOps lifecycle
  - Limitations
  - Risks
  - Hypotheses
  - Future work
  - References

- **Pitch** of up to 10 minutes followed by 5 minutes of questions, with:
  - Versioned slides in PDF or open format
  - Script covering problem/approach/demonstration/evidence/risks/governance/impact

- **Demonstration** of the platform, live or recorded, is desirable and adds bonus points:
  - Group should indicate the scenario
  - Register contingency plan
  - Version the recording or demonstration dataset

- **Pitch coverage** of presentation criteria:
  - **FinOps**: ROI, cost per Azure service, TCO
  - **Technical architecture justification** with diagram, boundaries, and discarded alternatives
  - **Scaling and reduction scenarios** by request volume

- Plan for periodic review of model card and system card with defined owners and cadence.

#### Acceptance Evidence

The evaluation panel finds coherent narrative of problem, solution, evidence, risks, governance, and business value, without claiming readiness for real regulated production.

---

## Evaluation Criteria

Evaluation follows the Phase 05 contract:

| Dimension | Weight | What the Panel Looks For |
|-----------|--------|-------------------------|
| **Business Criteria** | 30% | Adherence to chosen problem, impact clarity, viability, executive communication |
| **Global Technical Validation** | 70% | Pipeline, MLOps, evaluation, observability, security, governance, documentation, and PyTorch/MLflow use when applicable |

Groups must define specific metrics for the challenge. These metrics must be justified in the technical report and connected to business impact, but do not replace the official phase criteria.

---

## Pre-Demo Day Checklist

- ☐ README explains challenge, local execution, and limitations; pipeline uses compatible Kaggle dataset with source, version, license, and limitations.

- ☐ Processed dataset and synthetic enrichment documented and separated from original Kaggle dataset; experiments tracked in MLflow.

- ☐ Baseline and main approach compared with justified metrics; analysis references Thompson Sampling and lil'UCB.

- ☐ Evaluation includes golden set with at least 20 examples; guardrails tested with adversarial scenarios.

- ☐ Retraining, testing, approval, and promotion layer for new policies documented.

- ☐ Service, API, executable notebook, or demonstrable interface works with clear instructions and auditable decision log.

- ☐ Target architecture and deployment plan use exclusively Azure services, with secrets plan via Key Vault and Managed Identity.

- ☐ Model Card, System Card, and LGPD plan complete.

- ☐ Pitch separates problem, approach, demonstration, evidence, risks, and impact.

- ☐ Pitch covers FinOps (ROI, qualitative cost per Azure service, TCO).

- ☐ Pitch justifies technical architecture with diagram, boundaries, discarded alternatives, and presents scaling and reduction scenarios.

- ☐ Pitch includes live or recorded demonstration of the platform in operation, with contingency plan (desirable; adds bonus points).
