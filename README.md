# Datathon FIAP — Thompson Sampling Multi-Armed Bandit

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-blue)](https://mlflow.org/)
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
| Baseline (mix histórico, sem contexto) | 11,27% |
| **Thompson Sampling (sem contexto, Etapa 3)** | **14,97% (+3,70 p.p. vs. baseline)** |
| Melhor contexto | Senior + Other = 42,20% |
| Spread de conversão | 33,5 p.p. (~4,9x melhor que pior) |
| Contextos únicos | 12 (4 age_groups × 3 job_categories) |
| Testes automatizados | 60/60 passando |

---

## Dados e tratamento

**Base:** [bank-marketing (henriqueyamahata) — Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing) · arquivo `bank-additional-full.csv` · 41.188 contatos de campanhas de telemarketing bancário · alvo `y` = cliente assinou o depósito a prazo.

### Coluna removida por vazamento temporal: `duration`

`duration` é a duração em segundos da ligação — e é a **única** coluna descartada do dataset (41.188 linhas preservadas, 21 colunas → 20). O descarte acontece no ETL, em `src/datathon/etl/bank_marketing_primary.py` (`LEAKAGE_COLS = ["duration"]`), antes de qualquer parquet ser gravado. Os quatro ETLs do projeto aplicam a mesma regra.

Por que ela precisa sair, medido na base bruta:

| Medida | Valor |
|--------|-------|
| Duração média da ligação — cliente que **não** converteu | 220,8 s |
| Duração média da ligação — cliente que **converteu** | 553,2 s |
| Correlação `duration` × `y` | 0,405 |
| Conversão quando a ligação passou de 500 s | **42,57%** |
| Conversão quando a ligação durou menos de 100 s | **0,79%** |

Uma ligação longa não *causa* a conversão: ela é consequência do cliente já estar interessado. A coluna só passa a existir **depois** que a ligação termina, e a decisão de qual oferta apresentar é tomada **antes** dela começar. Um modelo treinado com `duration` teria métrica excelente no papel e seria inútil em produção, porque no momento da decisão esse valor não existe. Manter a coluna seria vazamento temporal — o caso citado explicitamente no enunciado do datathon.

Nenhuma outra coluna foi descartada. O restante do tratamento é apenas normalização: `y` convertido de `yes`/`no` para 1/0 e remoção de espaços nas colunas de texto.

---

## Instalação e execução

### Requisitos
- Python 3.11 ou superior
- pip
- Conta no Kaggle (os dados brutos vêm de lá)

### Caminho mínimo

Sete passos, na ordem. Rode todos a partir da raiz do repositório — os módulos de ETL resolvem `data/` em relação ao diretório atual.

**1. Clone o repositório**

```bash
git clone https://github.com/jemaldonado/FIAP_datathon_G37.git
cd FIAP_datathon_G37
```

**2. Crie e ative o ambiente virtual**

```
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

```

**3. Instale as dependências**

```
pip install -r requirements.txt

```

**4. Configure as credenciais do Kaggle**

Copie `.env.example` para `.env` e preencha os dois valores:

```
# Windows (PowerShell)
copy .env.example .env

# Linux / macOS
cp .env.example .env

```

```
KAGGLE_USERNAME=seu_usuario_kaggle
KAGGLE_KEY=sua_chave_de_api

```

Os dois valores estão no `kaggle.json` que o Kaggle baixa em **Settings → API → Create New Token**. O `.env` está no `.gitignore` e nunca deve ser commitado.

**5. Baixe os dados e gere a camada processada**

```
python scripts/download_data.py

```

Baixa as 4 bases para `data/kaggle/` e grava os parquets em `data/processed/`. Bases já baixadas são puladas; use `--force` para baixar de novo. Se preferir baixar manualmente, coloque o CSV em `data/kaggle/bank-marketing/` e rode o script — ele pula o download e vai direto para o ETL.

**6. Treine o modelo**

```
python scripts/train_simple.py

```

Grava `data/models/thompson_model.json`.

**7. Suba a API e o Dashboard**

Em terminais separados:

```
python src/datathon/api/app.py
streamlit run Painel_Datathon.py

```

A API estará em [http://localhost:5000/apidocs](http://localhost:5000/apidocs) e o Dashboard em [http://localhost:8501](http://localhost:8501/).

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

A base real só tem 2 canais de contato: `cellular` e `telephone`. Os 4 braços são esses 2 canais divididos por primeiro-contato vs. contato-repetido (`campaign == 1` ou não), relabeled com nomes de negócio (ver `datathon.bandit.assign_arm`):

```
Arm 0: Cellular_Standard  = cellular, primeiro contato
Arm 1: Email_Campaign     = cellular, contato repetido
Arm 2: SMS_Alert          = telephone, primeiro contato
Arm 3: Call_Premium       = telephone, contato repetido

```

Toda conversão contada por braço é 100% real (`y` observado), não são taxas estimadas ou simuladas. Só os nomes dos braços são uma convenção de negócio para dar 4 "estratégias" a um bandit contextual sobre uma base com só 2 canais reais. A interface "Thompson Aprende?" no dashboard mostra como o modelo aprende qual segmento converte melhor para cada contexto usando esses dados reais.

## Evidências de Deploy (Links Públicos AWS)

A aplicação encontra-se provisionada na nuvem pública da AWS (região `us-east-1`), operando em instâncias isoladas no cluster ECS, comunicando-se via variáveis de ambiente.

- **API (Back-end / Swagger):** `http://3.235.136.3:5000/apidocs`
- **Dashboard (Front-end / MLflow):** `http://34.226.194.199:8501/`

## Arquitetura

### Local (desenvolvimento)

```
Máquina local
├── Python Scripts
├── MLflow SQLite
├── Flask API (5000)
└── Painel Datathon (Streamlit - 8501)

```

### AWS (produção) — Implementado via Contêineres

```
Amazon ECR (Imagens Docker da API e Dashboard)
S3 (dados + modelo)
├── AWS Lambda (treino batch)
├── Amazon ECS Fargate (API Flask)
├── Amazon ECS Fargate (dashboard Streamlit)
└── CloudWatch (logs)

```

*Nota arquitetural:* A arquitetura original previa a utilização do AWS App Runner. No entanto, como a AWS decidiu encerrar o App Runner para novos clientes, o projeto foi modernizado utilizando o serviço substituto oficial, **Amazon ECS Express Mode (Fargate)**.

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

Dados e modelo são pequenos (KBs) e o treino roda em segundos — por isso a arquitetura é serverless/PaaS enxuta, sem clusters de treino gerenciados, que seriam desproporcionais ao tamanho real do problema.

Detalhes: [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) — desenho completo com custos por serviço em [`docs/architecture/AWS.md`](docs/architecture/AWS.md), [`docs/architecture/AZURE.md`](docs/architecture/AZURE.md) e [`docs/architecture/GCP.md`](docs/architecture/GCP.md).

## Golden set — 5 perfis validados

Thompson Sampling aprende diferenças por contexto. O golden set mostra o ranking das 4 estratégias para cada perfil:

| Perfil | Contexto | Melhor arm | Conversão | 2º melhor | 3º melhor |
|--------|----------|-----------|-----------|-----------|-----------|
| Jovem Técnico | Young + Technical | Cellular_Standard | 19,6% | Email (16,9%) | SMS (6,5%) |
| Executivo Prime | Prime + Business | Cellular_Standard | 15,0% | Email (11,2%) | SMS (7,4%) |
| Aposentado Senior | Senior + Other | Cellular_Standard | 48,4% | Email (42,3%) | Premium (23,8%) |
| Técnico Maduro | Mature + Technical | Cellular_Standard | 12,7% | Email (10,3%) | Premium (5,1%) |
| Profissional em Transição | Prime + Other | Email_Campaign | 15,9% | Cellular (14,1%) | SMS (5,4%) |

Mesmo para o melhor contexto (Senior+Other=48,4%), Email é 2º lugar com 42,3%. Thompson não escolhe Email como melhor porque Cellular ainda é superior globalmente. Já no contexto Prime+Other, Email de fato vence — mostrando que a recomendação muda de verdade por contexto, não é sempre "Cellular ganha".

Arquivo: `data/golden_set/golden_set.json`

## Como Thompson aprende com braços relabeled

### O dilema

A base real tem apenas 2 canais (cellular, telephone), mas Thompson Sampling precisa de múltiplas estratégias diferentes para aprender qual é melhor por contexto.

### A solução: relabel de segmentos reais, não estimativa

Os 2 canais reais são divididos por primeiro-contato vs. contato-repetido, dando 4 segmentos reais com nome de negócio:

| Arm | Segmento real | Taxa (base completa) |
|-----|------|------|
| Cellular_Standard | cellular, primeiro contato | 22,45% (contexto Young+Technical) |
| Email_Campaign | cellular, contato repetido | 18,72% (contexto Young+Technical) |
| SMS_Alert | telephone, primeiro contato | 7,88% (contexto Young+Technical) |
| Call_Premium | telephone, contato repetido | 4,65% (contexto Young+Technical) |

Todas as 4 taxas são medidas direto da base — não há estimativa nem justificativa de "quanto melhor/pior" um canal hipotético seria.

### Como Thompson aprende, na prática

```
FASE 1: Exploração (primeiros 100 clientes Young+Technical)
  Testa cada braço ~25 vezes, cada resultado é uma conversão real observada.

APRENDIZADO: o braço com mais sucessos reais até agora ganha mais tráfego.

FASE 2: Refinamento (próximos 900)
  Thompson ganha confiança na ordenação real: Cellular > Email > SMS > Call_Premium
  (para este contexto específico).

```

Conclusão: Thompson Sampling é robusto a erros nas estimativas iniciais.

## Testes automatizados

60 testes com pytest, todos passando.

```
pytest tests/ -v
# ======================== 60 passed in 2.15s ========================

```

Cobertura:

- `tests/test_api_endpoints.py` (17 testes): Integração e chamadas diretas.
- `tests/test_contextual_thompson.py` (23 testes): Lógica e serialização do modelo.
- `tests/test_canary_endpoints.py` (20 testes): Deploy seguro e regressão de aliasing.

## API REST

### Fazer recomendação

```
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

```
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

## Dashboard Interativo (Streamlit)

Todo o ecossistema visual da aplicação foi consolidado em uma **única interface multipágina**. Isso garante melhor navegação e evita múltiplas portas/processos espalhados.

Para executar o painel localmente:

```
streamlit run Painel_Datathon.py

```

### Navegação (Menu Lateral)

A barra lateral do Streamlit lista quatro entradas — a página principal e as três telas em `pages/`:

1. **Painel Datathon** (`Painel_Datathon.py`) — qualidade de dados e rigor estatístico. Submenu de rádio com: `Visão Geral`, `Qualidade de Dados` (validação ao vivo, nulos, duplicatas, alertas), `Métricas do Modelo` (12 contextos com intervalo de confiança), `Análise Estatística` (chi-square e calculadora de tamanho de amostra), `Demo da API` (formulário de recomendação + os 5 perfis do golden set) e `Canary Deploy` (documentação dos endpoints `/canary/*`).
2. **Canary Demo** (`pages/02_Canary_Demo.py`) — o caso real de troca de braço no segmento `Young_Technical`: o modelo treinado com os primeiros 70% dos contatos, em ordem cronológica real, recomenda `Email_Campaign` (10,72%); retreinado com todos os dados, passa a recomendar `Cellular_Standard` (19,66%). A comparação no topo lê os dois modelos do disco; a simulação ao vivo abaixo dela precisa da API rodando.
3. **Pipeline Completo** (`pages/03_Pipeline_Completo.py`) — visão ponta a ponta. Submenu com: `Overview`, `Dados` (treino/teste e heatmap de conversão idade × profissão), `Modelo`, `Thompson Aprende?`, `Golden Set`, `API` e `Métricas`.
4. **MLflow Showcase** (`pages/04_MLflow_Showcase.py`) — leitura direta de `.mlflow/mlflow.db` para exibir as runs reais, o baseline (11,27%) contra o Thompson Sampling (14,97%) e a análise de exploração por braço. Evidência da Etapa 7.

## Estrutura Atualizada

```
FIAP_datathon_G37/
├── README.md
├── requirements.txt                     Dependências
├── Dockerfile.api                       Build da Imagem Back-end
├── Dockerfile.dash                      Build da Imagem Front-end
├── docker-compose.yml                   Orquestração Local
├── Painel_Datathon.py                   App Central do Streamlit
├── pages/                               Telas filhas do Dashboard
├── src/datathon/
│   ├── config.py                        Config LOCAL/AWS/AZURE
│   ├── bandit/
│   │   └── contextual_thompson.py       Algoritmo
│   └── api/
│       └── app.py                       Flask API
├── scripts/
│   ├── train_simple.py                  Treino básico
│   └── train_with_mlflow.py             Treino + MLflow
├── tests/                               Testes Automatizados (Pytest)
├── data/                                Parquets e JSONs gerados
├── docs/                                Documentação Técnica Detalhada
└── .mlflow/                             Tracking de MLflow (SQLite)

```

## Resultados por contexto

| Posição | Contexto | Conversão | Clientes | Observação |
|---------|----------|-----------|----------|----------|
| 1 | Senior + Other | 42,2% | 891 | alvo prioritário |
| 2 | Senior + Technical | 32,5% | 206 | seniors muito valiosos |
| 3 | Young + Other | 30,3% | 891 | jovens fora do mercado de trabalho formal |
| ... | ... | ... | ... | ... |
| 12 | Mature + Business | 8,7% | 2.079 | evitar |

## Tecnologias

- Python 3.11 — linguagem principal
- Flask 3.0 — framework REST
- Streamlit — framework de interface interativa
- MLflow 2.0 — experiment tracking
- pandas/numpy — processamento de dados
- pytest — testes automatizados
- Docker & Docker Compose — encapsulamento e padronização
- Amazon ECS (Fargate) + ECR + S3 — deploy em nuvem principal

## Contato

- Desenvolvido por: Grupo 37 - Datathon FIAP
- Repositório: https://github.com/jemaldonado/FIAP_datathon_G37
- Dados: [bank-marketing (henriqueyamahata) — Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)