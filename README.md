# Datathon FIAP — Thompson Sampling Multi-Armed Bandit

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-blue)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/Tests-60%2F60%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#license)

Recomendador de canal de contato bancário via Thompson Sampling contextual (age_group × job_category).

---

## Resumo

Este projeto implementa um multi-armed bandit contextual para otimizar estratégias de contato em campanhas de marketing bancário. O sistema aprende qual estratégia funciona melhor para cada perfil de cliente e expõe a recomendação via API REST.

### Resultados principais

| Métrica | Resultado |
|---------|-----------|
| Clientes analisados | 41.188 |
| Taxa de conversão geral | 11,27% |
| Melhor contexto | Senior + Other = 42,20% |
| Spread de conversão | 33,5 p.p. (~4,9x melhor que pior) |
| Contextos únicos | 12 (4 age_groups × 3 job_categories) |
| Testes automatizados | 60/60 passando |

---

## Instalação e execução

### Requisitos
- Python 3.11+
- pip ou uv
- Conta Kaggle (para baixar os dados)

### Passos

```bash
# 1. Clone
git clone https://github.com/jemaldonado/FIAP_datathon_G37.git
cd FIAP_datathon_G37

# 2. Instale dependências
pip install -r requirements.txt

# 3. Baixe dados (automático ou manual)
python scripts/download_data.py

# 4. Treinar modelo
python scripts/train_simple.py

# 5. Rodar API
python src/datathon/api/app.py

# 6. Testar
# Abra http://localhost:5000/apidocs no navegador
```

---

## Como funciona

### Thompson Sampling, resumido

Thompson Sampling resolve o dilema exploração vs. explotação:

```
Para cada cliente:
1. Identifica o contexto (idade, profissão)
2. Usa Thompson Sampling para esse contexto
3. Recomenda a estratégia com maior conversão esperada
4. Aprende com o resultado (converteu? atualiza a distribuição)
```

### 4 estratégias de campanha (arms)

A base real só tem 2 canais de contato: `cellular` e `telephone`. Os 4 braços são esses 2
canais divididos por primeiro-contato vs. contato-repetido (`campaign == 1` ou não),
relabeled com nomes de negócio (ver `datathon.bandit.assign_arm`):

```
Arm 0: Cellular_Standard  = cellular, primeiro contato
Arm 1: Email_Campaign     = cellular, contato repetido
Arm 2: SMS_Alert          = telephone, primeiro contato
Arm 3: Call_Premium       = telephone, contato repetido
```

Toda conversão contada por braço é 100% real (`y` observado), não são taxas estimadas ou
simuladas. Só os nomes dos braços são uma convenção de negócio para dar 4 "estratégias" a um
bandit contextual sobre uma base com só 2 canais reais. A página "Thompson Aprende?" no
dashboard mostra como o modelo aprende qual segmento converte melhor para cada contexto usando
esses dados reais.

### 12 contextos (age_group × job_category)

```
Age Groups:       Job Categories:    Total Contextos:
- Young (18-30)   - Technical         = 12 (4 × 3)
- Prime (30-45)   - Business
- Mature (45-60)  - Other
- Senior (60+)
```

---

## Arquitetura

### Local (desenvolvimento)
```
Máquina local
├── Python Scripts
├── MLflow SQLite
├── Flask API (5000)
└── Swagger UI
```

### AWS (produção)
```
S3 (dados + modelo)
├── Lambda (treino batch)
├── App Runner (API Flask)
├── App Runner (dashboard Streamlit)
└── CloudWatch (logs)
```

### Azure (produção)
```
Blob Storage (dados + modelo)
├── Azure Functions (treino batch)
├── App Service (API Flask)
├── App Service (dashboard Streamlit)
└── Application Insights (logs)
```

### GCP (produção)
```
Cloud Storage (dados + modelo)
├── Cloud Run Jobs (treino batch)
├── Cloud Run service (API Flask — traffic splitting nativo entre revisions)
├── Cloud Run service (dashboard Streamlit)
└── Cloud Logging + Cloud Monitoring (logs)
```

Dados e modelo são pequenos (KBs) e o treino roda em segundos — por isso a arquitetura é
serverless/PaaS enxuta, sem clusters de treino gerenciados (SageMaker Training Jobs, Azure ML
Compute Clusters, Vertex AI Training), que seriam desproporcionais ao tamanho real do problema.

Detalhes: [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) — desenho completo com custos por
serviço (fonte oficial + data de consulta) em [`docs/architecture/AWS.md`](docs/architecture/AWS.md),
[`docs/architecture/AZURE.md`](docs/architecture/AZURE.md) e [`docs/architecture/GCP.md`](docs/architecture/GCP.md).

---

## Golden set — 5 perfis validados

Thompson Sampling aprende diferenças por contexto. O golden set mostra o ranking das 4
estratégias para cada perfil:

| Perfil | Contexto | Melhor arm | Conversão | 2º melhor | 3º melhor |
|--------|----------|-----------|-----------|-----------|-----------|
| Jovem Técnico | Young + Technical | Cellular_Standard | 19,6% | Email (16,9%) | SMS (6,5%) |
| Executivo Prime | Prime + Business | Cellular_Standard | 15,0% | Email (11,2%) | SMS (7,4%) |
| Aposentado Senior | Senior + Other | Cellular_Standard | 48,4% | Email (42,3%) | Premium (23,8%) |
| Técnico Maduro | Mature + Technical | Cellular_Standard | 12,7% | Email (10,3%) | Premium (5,1%) |
| Profissional em Transição | Prime + Other | Email_Campaign | 15,9% | Cellular (14,1%) | SMS (5,4%) |

Mesmo para o melhor contexto (Senior+Other=48,4%), Email é 2º lugar com 42,3%. Thompson não
escolhe Email como melhor porque Cellular ainda é superior globalmente. Já no contexto
Prime+Other, Email de fato vence — mostrando que a recomendação muda de verdade por contexto,
não é sempre "Cellular ganha".

Arquivo: `data/golden_set/golden_set.json`

---

## Como Thompson aprende com braços relabeled

### O dilema
A base real tem apenas 2 canais (cellular, telephone), mas Thompson Sampling precisa de
múltiplas estratégias diferentes para aprender qual é melhor por contexto.

### A solução: relabel de segmentos reais, não estimativa

Os 2 canais reais são divididos por primeiro-contato vs. contato-repetido, dando 4 segmentos
reais com nome de negócio (ver `datathon.bandit.assign_arm`):

| Arm | Segmento real | Taxa (base completa) |
|-----|------|------|
| Cellular_Standard | cellular, primeiro contato | 22,45% (contexto Young+Technical) |
| Email_Campaign | cellular, contato repetido | 18,72% (contexto Young+Technical) |
| SMS_Alert | telephone, primeiro contato | 7,88% (contexto Young+Technical) |
| Call_Premium | telephone, contato repetido | 4,65% (contexto Young+Technical) |

Todas as 4 taxas são medidas direto da base — não há estimativa nem justificativa de "quanto
melhor/pior" um canal hipotético seria. A única convenção é o nome do braço.

### Como Thompson aprende, na prática

```
FASE 1: Exploração (primeiros 100 clientes Young+Technical)
  Testa cada braço ~25 vezes, cada resultado é uma conversão real observada.

APRENDIZADO: o braço com mais sucessos reais até agora ganha mais tráfego.

FASE 2: Refinamento (próximos 900)
  Thompson ganha confiança na ordenação real: Cellular > Email > SMS > Call_Premium
  (para este contexto específico — outros contextos têm ordenações diferentes,
  ver golden set acima).
```

Por depender só de conversões reais, a incerteza do Thompson Sampling não vem de "a taxa pode
estar errada" — vem de quantos clientes reais cada braço já teve nesse contexto (poucos
trials = posterior largo/incerto; muitos trials = posterior estreito/confiante).

### Validação no dashboard
A página "Thompson Aprende?" mostra:
1. Como Thompson explora/explota gradualmente (exemplo ilustrativo do mecanismo)
2. De onde vem a incerteza real (tamanho de amostra por braço, não estimativa)
3. Por que o relabel de segmentos reais é uma escolha defensável, e qual é a limitação real

Conclusão: Thompson Sampling é robusto a erros nas estimativas iniciais.

---

## Testes automatizados

60 testes com pytest, todos passando.

```bash
# Rodar testes
pytest tests/ -v

# Resultado:
# ======================== 60 passed in 2.15s ========================
```

Cobertura:
- `tests/test_api_endpoints.py` (17 testes): GET /health (3), POST /recommend (8),
  GET /swagger.json (4), integração (2)
- `tests/test_contextual_thompson.py` (23 testes): modelo, contextos, serialização, validação
  de dados
- `tests/test_canary_endpoints.py` (20 testes): `/canary/start` (7), `/canary/recommend` (4),
  `/canary/metrics` (3), `/canary/promote` e `/canary/rollback` (4), regressão do bug de
  aliasing entre modelo canary e modelo de produção (2)

---

## API REST

### Fazer recomendação

```bash
POST /recommend
Content-Type: application/json

{
  "age": 35,
  "job": "admin",
  "marital": "married",
  "education": "university.degree",
  "contact": "cellular",
  "campaign": 1
}
```

Resposta:
```json
{
  "recommended_arm": 0,
  "arm_name": "Cellular_Standard",
  "expected_conversion": 0.15,
  "context": {
    "age_group": "Prime",
    "job_category": "Technical"
  },
  "rationale": "Contextual Thompson Sampling for Prime + Technical recommends Cellular_Standard (15.0% historical conversion rate)."
}
```

### Endpoints disponíveis

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/` | API metadata |
| GET | `/health` | Health check |
| POST | `/recommend` | Recomendação |
| GET | `/swagger.json` | OpenAPI spec |
| GET | `/apidocs` | Swagger UI |
| POST | `/canary/start` | Inicia canary deploy (% de tráfego para modelo novo) |
| POST | `/canary/recommend` | Recomendação com roteamento canary/baseline |
| GET | `/canary/metrics` | Métricas comparativas + teste qui-quadrado |
| POST | `/canary/promote` | Promove canary para baseline |
| POST | `/canary/rollback` | Reverte para baseline |

Demo interativa do canary: `streamlit run app_canary_arm_swap_demo.py` (porta 8501) —
roteiro em [`CANARY_DEMO_GUIDE.md`](CANARY_DEMO_GUIDE.md) e explicação em
[`docs/CANARY_DEPLOY_EXPLAINED.md`](docs/CANARY_DEPLOY_EXPLAINED.md). Detalhes de todos os
dashboards na seção [Dashboards](#dashboards) abaixo.

---

## Estrutura

```
FIAP_datathon_G37/
├── README.md
├── requirements.txt                     Dependências
├── src/datathon/
│   ├── config.py                        Config LOCAL/AWS/AZURE
│   ├── bandit/
│   │   └── contextual_thompson.py       Algoritmo
│   └── api/
│       └── app.py                       Flask API
├── scripts/
│   ├── train_simple.py                  Treino básico
│   └── train_with_mlflow.py             Treino + MLflow
├── tests/
│   ├── test_contextual_thompson.py      Testes modelo
│   └── test_api_endpoints.py            Testes API (22)
├── data/
│   ├── processed/
│   │   └── bank_marketing_primary.parquet
│   ├── models/
│   │   └── thompson_model.json
│   └── golden_set/
│       └── golden_set.json
├── docs/
│   ├── ARQUITETURA.md
│   ├── LGPD_PLAN.md
│   ├── RETRAINING_PIPELINE.md
│   └── ROTEIRO_APRESENTACAO.md          Roteiro do vídeo pitch (Etapa 8)
└── .mlflow/
    └── mlflow.db                        Tracking MLflow
```

---

## Resultados por contexto

| Posição | Contexto | Conversão | Clientes | Observação |
|---------|----------|-----------|----------|----------|
| 1 | Senior + Other | 42,2% | 891 | alvo prioritário |
| 2 | Senior + Technical | 32,5% | 206 | seniors muito valiosos |
| 3 | Young + Other | 30,3% | 891 | jovens fora do mercado de trabalho formal |
| 4 | Senior + Business | 30,2% | 96 | seniors em geral melhores |
| ... | ... | ... | ... | ... |
| 12 | Mature + Business | 8,7% | 2.079 | evitar |

---

## Tecnologias

- Python 3.11 — linguagem principal
- Flask 3.0 — framework REST
- MLflow 2.0 — experiment tracking
- pandas/numpy — processamento de dados
- pytest — testes
- AWS S3 + Lambda + App Runner — deploy em nuvem (opcional)
- Azure Blob Storage + Functions + App Service — deploy em nuvem (opcional)
- GCP Cloud Storage + Cloud Run Jobs + Cloud Run — deploy em nuvem (opcional)

---

## Deploy em nuvem

### AWS
```bash
export DATATHON_ENV=aws
export AWS_ACCESS_KEY_ID=xxxxx
export AWS_SECRET_ACCESS_KEY=xxxxx
export AWS_S3_BUCKET=meu-bucket

python scripts/train_with_mlflow.py --env aws
```

### Azure
```bash
export DATATHON_ENV=azure
export AZURE_SUBSCRIPTION_ID=xxxxx
export AZURE_ML_WORKSPACE=meu-workspace

python scripts/train_with_mlflow.py --env azure
```

### GCP
```bash
export DATATHON_ENV=gcp
export GOOGLE_CLOUD_PROJECT=meu-projeto
export GCS_BUCKET=meu-bucket

python scripts/train_with_mlflow.py --env gcp
```

> `Environment.GCP` ainda não existe em `src/datathon/config.py` (só `LOCAL`/`AWS`/`AZURE`) — ver
> achado detalhado em [`docs/architecture/GCP.md`](docs/architecture/GCP.md) seção 5.5. O comando
> acima é o alvo depois dessa correção, não o comportamento atual.

Guia completo: [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md)

---

## Dataset

Primário: [Bank Marketing - UCI](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
- 41.188 clientes
- Campanhas de marketing de um banco português
- Variáveis: idade, profissão, saldo, contato prévio, etc.

---

## Status

Modelo Thompson Sampling treinado, API Flask com Swagger funcionando, 60 testes
automatizados passando, golden set com 5 perfis, tracking MLflow configurado, suporte
dual-environment (LOCAL/AWS/AZURE) e documentação de arquitetura concluída.

---

## Dashboards

Além da API (`/recommend`, `/canary/*` — ver seção [API REST](#api-rest)), o projeto tem
4 dashboards Streamlit, cada um com um propósito diferente. Qualquer um roda com
`streamlit run <arquivo>.py` (porta padrão 8501; se mais de um estiver aberto ao mesmo tempo,
o Streamlit sobe automaticamente para a próxima porta livre — 8502, 8503...).

### `app_canary_arm_swap_demo.py` — caso real de canary deploy

```bash
python src/datathon/api/app.py         # terminal 1 — precisa da API rodando
streamlit run app_canary_arm_swap_demo.py   # terminal 2
```

Mostra o caso real do segmento `Young_Technical`: um retreino com mais dados troca a oferta
vencedora (Email_Campaign → Cellular_Standard). Compara o modelo estaticamente e permite rodar
requisições reais ao vivo contra `/canary/start` / `/canary/recommend` para demonstrar a troca
de maioria. Roteiro detalhado em [`CANARY_DEMO_GUIDE.md`](CANARY_DEMO_GUIDE.md).

### `app_mlflow_showcase.py` — evidência do tracking MLflow

```bash
streamlit run app_mlflow_showcase.py
```

Não precisa da API. Lê direto de `.mlflow/mlflow.db` e do parquet processado: mostra as runs
reais registradas, o gráfico baseline vs. Thompson (Etapa 3, 11,27% → 14,97%) e a análise de
exploração por braço. Evidência para o item "tracking de experimentos via MLOps" do checklist.

### `app_dashboard.py` — pipeline completo (dados → treino → avaliação → API)

```bash
streamlit run app_dashboard.py
```

Não precisa da API. Visão geral de ponta a ponta do projeto: dados processados, treino do
modelo, avaliação por contexto e simulação de recomendações — útil para navegar o pipeline
inteiro numa tela só.

### `app_dashboard_pt.py` — qualidade de dados e métricas do modelo (PT)

```bash
streamlit run app_dashboard_pt.py
```

Não precisa da API para abrir (algumas seções mostram exemplos de `curl` contra os endpoints
de canary como documentação, sem chamá-los de fato). Painel em português com validação de
qualidade de dados ao vivo, métricas de desempenho, testes estatísticos (intervalos de
confiança, qui-quadrado) e a página "Thompson Aprende?", que explica de onde vem a
incerteza do modelo (tamanho de amostra por braço, não estimativa).

---

## Contato

- Desenvolvido por: Grupo 37 - Datathon FIAP
- Repositório: https://github.com/jemaldonado/FIAP_datathon_G37
- Dados: [Kaggle Dataset](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)

---

Última atualização: 2026-08-05 | Versão: 1.0.0
