# Relatório Técnico

**Datathon 7MLET · POS TECH / FIAP · Grupo 37**
**Produto:** recomendador de canal de campanha bancária via Thompson Sampling contextual

> Este relatório resume o essencial de cada etapa do edital e aponta para o artefato real
> correspondente no repositório. Nenhum número aqui é estimado à mão — vem de execução real do
> código (`pytest`, notebooks, MLflow) nesta data.

---

## 1. Problema

Uma instituição financeira digital precisa decidir, para cada cliente elegível, **qual canal de
contato oferecer** (ligação por celular ou por telefone fixo) para maximizar conversão em depósito
a prazo. Regras fixas e testes A/B longos desperdiçam tráfego no canal perdedor e não se adaptam a
diferenças entre segmentos de cliente.

**Solução construída:** um *multi-armed bandit* contextual (Thompson Sampling, priors
Beta-Bernoulli) que aprende, por segmento (`age_group` × `job_category`), qual estratégia de
contato converte melhor — comparado a uma regra fixa de baseline.

---

## 2. Dados (Etapas 1–2)

- **Base:** [`bank-marketing` (Kaggle, henriqueyamahata)](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
  — campanhas de telemarketing de um banco português, 41.188 clientes, alvo `y` (assinou depósito a
  prazo?).
- **Coluna de vazamento removida:** `duration` (só é conhecida depois da ligação, não pode ser
  feature de decisão).
- **EDA e tratamento:** [`notebooks/01_eda_contextual_bandits.ipynb`](notebooks/01_eda_contextual_bandits.ipynb).
- **Dado processado:** `data/processed/bank_marketing_primary.parquet` (171 KB).
- **Contextos derivados:** `age_group` (Young/Prime/Mature/Senior) × `job_category`
  (Technical/Business/Other) → 12 contextos.

---

## 3. Baseline e estratégia algorítmica (Etapa 3)

**Braços reais observados na base:** `cellular` (14,74% conversão) e `telephone` (5,23%
conversão) — únicos dois canais realmente presentes nos dados.

**Baseline (regra fixa):** taxa de conversão do mix histórico real (o que o banco de fato fez),
`df['y'].mean()` = **11,27%**.

**Algoritmo adaptativo:** Thompson Sampling Beta-Bernoulli. Cada rodada sorteia um braço via
posterior; a recompensa é simulada como `Bernoulli(taxa_real_do_braço)`, usando as taxas medidas
acima (nenhum número inventado). Reproduzível em
[`notebooks/datathon_main.ipynb`](notebooks/datathon_main.ipynb) e via
`compute_baseline_vs_thompson()` em [`src/datathon/bandit/contextual_thompson.py`](src/datathon/bandit/contextual_thompson.py).

| Métrica | Valor |
|---|---|
| Baseline | 11,27% |
| Thompson Sampling | 14,97% |
| **Melhoria** | **+3,70 p.p.** |
| Trials gastos explorando o braço pior (`telephone`) | 130 de 41.188 (0,3%) |

> **Nota de rigor:** a primeira versão desta simulação tinha um bug — a recompensa não dependia do
> braço escolhido, então Thompson e baseline davam sempre o mesmo número por construção. Corrigido
> em 2026-08-05; ver histórico de commits e `app_mlflow_showcase.py` para a run anterior (quebrada)
> mantida no MLflow por transparência.

---

## 4. Bandit contextual em produção (Etapa 3, extensão)

Para produção (API, dashboard, canary), o projeto usa `ContextualThompsonSampling` — 12 bandits
Beta-Bernoulli independentes (um por contexto) e **4 braços**. A base real só tem 2 canais de
contato (`contact`: cellular/telephone); os 4 braços são esses 2 canais divididos por
primeiro-contato vs. contato-repetido (`campaign == 1` ou não) e relabeled com nomes de negócio.
Toda conversão contada por braço é 100% real — não há taxa estimada ou simulada:

| Braço | Segmento real | Origem |
|---|---|---|
| Cellular_Standard | cellular, primeiro contato | `contact=='cellular' & campaign==1` |
| Email_Campaign | cellular, contato repetido | `contact=='cellular' & campaign>1` |
| SMS_Alert | telephone, primeiro contato | `contact=='telephone' & campaign==1` |
| Call_Premium | telephone, contato repetido | `contact=='telephone' & campaign>1` |

Ver a regra completa, documentada, em `datathon.bandit.assign_arm` (usada por
`scripts/retrain_model.py`, o pipeline real — não em `scripts/train_model.py`, um script
exploratório/legado com uma regra estocástica diferente, nunca usado em produção) e
[`README.md`](README.md#como-thompson-aprende-com-braços-relabeled). Este modelo de 4 braços é
o que roda na API e no dashboard — a comparação de 2 braços reais da seção 3 é a evidência exigida
pela Etapa 3/7 (baseline vs adaptativo com dado 100% real).

> **Nota de rigor (auditoria 2026-08-05):** uma versão anterior deste relatório descrevia
> Email_Campaign/SMS_Alert/Call_Premium como "sintéticos, taxa estimada" — isso descrevia a
> metodologia do script legado `train_model.py`, não o que roda de fato em produção. Corrigido
> depois de uma auditoria de código encontrar a divergência entre os dois scripts.

---

## 5. Avaliação e casos de teste (Etapa 4)

- **Golden Set** — 5 perfis de cliente com ranking das 4 estratégias por perfil, em
  `data/golden_set/golden_set.json` (ver tabela em [`README.md`](README.md#golden-set--5-perfis-validados)).
- **Testes automatizados:** 60 testes `pytest`, todos passando —
  `tests/test_contextual_thompson.py` (23: modelo, contextos, serialização, validação de dados),
  `tests/test_api_endpoints.py` (17: health, `/recommend`, swagger, integração) e
  `tests/test_canary_endpoints.py` (20: `/canary/start`, `/canary/recommend`, `/canary/metrics`,
  `/canary/promote`/`/canary/rollback`, incluindo regressão do bug de aliasing entre o modelo
  canary e o modelo de produção encontrado pela auditoria de 2026-08-05).

```bash
pytest tests/ -v
```

---

## 6. Serviço demonstrável (Etapa 5)

- **API Flask:** [`src/datathon/api/app.py`](src/datathon/api/app.py) — `POST /recommend` recebe
  perfil do cliente e retorna braço recomendado + taxa esperada + contexto. Swagger em
  `/apidocs`.
- **Dashboards Streamlit:** `app_dashboard_pt.py` (visão geral do modelo),
  `app_canary_arm_swap_demo.py` (canary deploy), `app_mlflow_showcase.py`
  (evidência do tracking MLflow, seção 8), `app_dashboard.py` (visão do pipeline completo:
  dados → treino → avaliação → API). Detalhes de instalação/uso de cada um em
  [`README.md`](README.md#dashboards).

---

## 7. Arquitetura-alvo em nuvem (Etapa 6)

Sistema pequeno por natureza (modelo ~5 KB, parquet 171 KB, treino em segundos) → arquitetura-alvo
serverless/PaaS enxuta, sem cluster de treino gerenciado (SageMaker Training Jobs, Azure ML Compute
Clusters, Vertex AI Training seriam desproporcionais).

- Visão comparativa AWS × Azure × GCP: [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md)
- Desenho detalhado com custo por serviço (fonte oficial + data de consulta):
  [`docs/architecture/AWS.md`](docs/architecture/AWS.md) ·
  [`docs/architecture/AZURE.md`](docs/architecture/AZURE.md) ·
  [`docs/architecture/GCP.md`](docs/architecture/GCP.md) — este último com destaque para o
  *traffic splitting* nativo do Cloud Run como o mecanismo de canary deploy mais direto das três
  nuvens avaliadas

---

## 8. Ciclo de vida MLOps (Etapa 7)

- **Tracking:** MLflow local (SQLite, `.mlflow/mlflow.db`), experimento
  `Datathon-Thompson-Sampling`. Params (`n_customers`, `n_arms`, `n_contexts`, `algorithm`) e
  métricas — incluindo `baseline_rate`, `thompson_rate_etapa3`, `improvement_vs_baseline`,
  `beat_baseline` (seção 3) — gravados por `scripts/train_with_mlflow.py` e
  `scripts/retrain_model.py`.
- **Evidência navegável dos resultados reais:** `streamlit run app_mlflow_showcase.py`.
- **Retreino versionado:** [`docs/RETRAINING_PIPELINE.md`](docs/RETRAINING_PIPELINE.md),
  `scripts/retrain_model.py`.
- **Canary deploy:** [`docs/CANARY_DEPLOY_EXPLAINED.md`](docs/CANARY_DEPLOY_EXPLAINED.md). Caso
  real de troca de braço vencedor após retreino: contexto `Young_Technical`
  (idade 18–29, cargo admin/technician/engineer/scientist/services) — um snapshot treinado
  com os primeiros 70% dos contatos em ordem cronológica real recomenda `Email_Campaign`
  (11,67%), o candidato treinado com todos os dados disponíveis recomenda `Cellular_Standard`
  (22,51%). O corte é temporal (ordem real de contato, ver `--temporal-cutoff` em
  `scripts/retrain_model.py`), não uma amostra aleatória embaralhada — uma versão anterior
  desta demo usava reamostragem aleatória da mesma base, o que é ruído, não dado novo (corrigido
  após auditoria 2026-08-05). Demonstrável com `python scripts/demo_arm_swap_case.py` ou
  `streamlit run app_canary_arm_swap_demo.py`.
- **Qualidade de dados / logs:** [`docs/DATA_QUALITY_E_LOGS.md`](docs/DATA_QUALITY_E_LOGS.md).

---

## 9. Governança e LGPD

Nenhum dado pessoal real de cliente é usado — base pública anonimizada (Kaggle). Ver postura
completa em [`docs/LGPD_PLAN.md`](docs/LGPD_PLAN.md), incluindo a distinção entre os braços
relabeled em produção (conversão 100% real, ver seção 4) e o exercício exploratório separado
e não usado (`campaign_synthesis.py`), que de fato simula taxas hipotéticas.

---

## 10. Limitações conhecidas

- Os 4 braços do modelo de produção são 100% observados na base real (seção 4) — mas conflam
  canal (cellular/telephone) com posição no contato (primeiro vs. repetido). Não há como
  isolar, só com este dataset, se a diferença de conversão vem do canal em si ou de o cliente já
  ter sido contatado antes.
- `telephone` nunca é o braço vencedor em nenhum dos 12 contextos reais (só `Cellular_Standard`
  ou `Email_Campaign`, ambos derivados do canal `cellular`) — a heterogeneidade contextual do
  projeto vem da diferença real de conversão entre os canais cellular e telephone nesta base,
  não de estimativa alguma.
- Arquitetura de nuvem (Etapa 6) é um desenho, não uma implantação real — ver nota de método em
  cada documento de `docs/architecture/`.

---

## Referências

- Edital: [`plan/Datathon-7MLET.md`](plan/Datathon-7MLET.md)
- Visão geral e quick start: [`README.md`](README.md)
- Arquitetura: [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md), [`docs/architecture/AWS.md`](docs/architecture/AWS.md), [`docs/architecture/AZURE.md`](docs/architecture/AZURE.md), [`docs/architecture/GCP.md`](docs/architecture/GCP.md)
- MLOps: [`docs/RETRAINING_PIPELINE.md`](docs/RETRAINING_PIPELINE.md), [`docs/CANARY_DEPLOY_EXPLAINED.md`](docs/CANARY_DEPLOY_EXPLAINED.md)
- Qualidade de dados: [`docs/DATA_QUALITY_E_LOGS.md`](docs/DATA_QUALITY_E_LOGS.md)
- Governança: [`docs/LGPD_PLAN.md`](docs/LGPD_PLAN.md)
- Roteiro de apresentação: [`docs/ROTEIRO_APRESENTACAO.md`](docs/ROTEIRO_APRESENTACAO.md)
