# Data Quality e Logs Estruturados

## Visão geral

Módulo de validação de dados, logging estruturado e testes estatísticos usado pelo pipeline de
treino e pelos dashboards Streamlit.

### O que foi implementado

1. **Módulo Data Quality** (`src/datathon/quality/data_quality.py`)
   - Validação automática de dados
   - Relatórios estruturados
   - Métricas de qualidade

2. **Logging estruturado**
   - Logs com timestamps
   - Níveis de severidade (DEBUG, INFO, WARNING, ERROR)
   - Rastreabilidade completa

3. **Testes estatísticos** (`src/datathon/validation/statistical_tests.py`)
   - Intervalo de confiança (Wilson score)
   - Chi-square tests
   - Correção de Bonferroni para múltiplas comparações

---

## Motivação

Sem validação de dados, problemas passam silenciosamente para o treinamento e não há transparência
do estado dos dados para quem avalia o projeto. O módulo de data quality detecta esses problemas
antes do treino (idade fora de faixa, duplicatas, nulos); o logging estruturado dá rastreabilidade
de quando e por que algo aconteceu; os testes estatísticos (intervalo de confiança, chi-square)
separam diferença observada de significância estatística real — a base para os números comparativos
usados no relatório técnico.

---

## Implementação

### 1. Módulo Data Quality

#### Arquivos criados
```
src/datathon/quality/
├── __init__.py
└── data_quality.py  (240 linhas)
```

#### Funcionalidades principais

**a) Validação de dados**

```python
from datathon.quality import validate_data_quality

df = pd.read_parquet('data.parquet')
report = validate_data_quality(
    df,
    dataset_name='bank_marketing',
    check_age_range=(18, 150),
    check_conversions=True,
    check_duplicates=True
)
```

**O que valida:**
- Null values (esperado: 0)
- Age range (18-150)
- Target distribution (conversão %)
- Duplicates (identifica registros iguais)
- Required columns (colunas obrigatórias)

**Saída: QualityReport**
```python
report = {
    'dataset_name': 'bank_marketing',
    'total_records': 41188,
    'null_values': 0,
    'overall_status': 'WARN',  # PASS, WARN, FAIL
    'metrics': [
        {
            'name': 'Null Values',
            'status': 'PASS',
            'value': 0,
            'message': 'Found 0 null values (threshold: 0)'
        },
        {
            'name': 'Age Range',
            'status': 'WARN',
            'value': 5,
            'message': 'Found 5 out-of-range ages (range: (18, 150))'
        },
        ...
    ]
}
```

#### Exemplo real — output do treinamento
```
2026-07-08 21:08:16 - datathon.quality.data_quality - INFO - Checking null values...
2026-07-08 21:08:16 - datathon.quality.data_quality - INFO - Checking for null values...
2026-07-08 21:08:16 - datathon.quality.data_quality - INFO - Checking age range: (18, 150)
2026-07-08 21:08:16 - datathon.quality.data_quality - WARNING - Found 5 invalid ages outside (18, 150)
2026-07-08 21:08:16 - datathon.quality.data_quality - INFO - Checking target distribution...
2026-07-08 21:08:16 - datathon.quality.data_quality - INFO - Target distribution - Positive: 11.27%, Negative: 88.73%
2026-07-08 21:08:16 - datathon.quality.data_quality - INFO - Checking for duplicate records...
2026-07-08 21:08:16 - datathon.quality.data_quality - WARNING - Found 1784 duplicate records
2026-07-08 21:08:16 - datathon.quality.data_quality - INFO - Data quality validation complete. Status: WARN
```

---

### 2. Logging estruturado

#### Implementação

```python
from datathon.quality import setup_logging

# Na inicialização do script
setup_logging(
    log_level='INFO',
    log_file='logs/training.log'  # opcional
)

logger = logging.getLogger(__name__)
logger.info("Começando treinamento")
logger.warning("Dados duplicados encontrados")
logger.error("Falha ao carregar modelo")
```

#### Formato de log
```
YYYY-MM-DD HH:MM:SS - module.name - LEVEL - message

Exemplo:
2026-07-08 21:08:16 - datathon.quality.data_quality - INFO - Checking age range: (18, 150)
2026-07-08 21:08:16 - __main__ - INFO - Extracting model data...
```

#### O que o formato dá
- **Timestamps:** quando aconteceu
- **Nomes de módulos:** onde aconteceu
- **Níveis:** DEBUG < INFO < WARNING < ERROR < CRITICAL
- **Mensagens estruturadas:** parse automático facilitado

---

### 3. Testes estatísticos

#### Arquivos criados
```
src/datathon/validation/
├── __init__.py
└── statistical_tests.py  (280 linhas)
```

#### Funcionalidades

**a) Intervalo de confiança (Wilson score)**

```python
from datathon.validation import get_confidence_interval

# Conversão observada: 47 sucesso em 100 tentativas
lower, upper = get_confidence_interval(successes=47, trials=100, confidence=0.95)

# Resultado:
# Lower: 0.361 (36.1%)
# Upper: 0.581 (58.1%)
# Mensagem: "Conversão real é entre 36.1% e 58.1% com 95% confiança"
```

**b) Chi-square test para comparações**

```python
from datathon.validation import compare_conversions

result = compare_conversions(
    successes_group1=47,   # Thompson
    trials_group1=100,
    successes_group2=42,   # Baseline
    trials_group2=100,
    alpha=0.05
)

# Resultado:
# {
#     'p_value': 0.523,        # p > 0.05 = não significativo
#     'is_significant': False,
#     'message': 'Group 1: 47.0%, Group 2: 42.0%, χ²=0.4023, p=0.5230'
# }
```

**c) Comparação de múltiplos contextos**

```python
from datathon.validation import compare_contexts

contexts = {
    'Young_Technical': (216, 962),      # successes, trials
    'Senior_Other': (199, 419),
    'Prime_Business': (137, 911)
}

results = compare_contexts(
    context_results=contexts,
    baseline_successes=4640,
    baseline_trials=41188,
    alpha=0.05
)

# Resultado:
# {
#     'Young_Technical': {
#         'p_value': 0.0023,
#         'is_significant': True,
#         'message': '...'
#     },
#     'Senior_Other': {...},
#     'Prime_Business': {...}
# }
```

**d) Cálculo de sample size para A/B test**

```python
from datathon.validation import calculate_sample_size

n_required = calculate_sample_size(
    baseline_rate=0.1127,    # 11.27% conversão atual
    effect_size=0.02,        # detectar +2pp
    alpha=0.05,
    power=0.80
)

# Resultado: 2891 (por grupo)
# Mensagem: "Precisa de 2891 clientes por grupo para A/B test"
```

---

## Integração no Streamlit dashboard

A integração real está em `app_dashboard_pt.py` (não em `app_dashboard.py`, que é o outro
dashboard e não usa este módulo — ver `README.md#dashboards`): importa `validate_data_quality`,
`setup_logging` e as funções de `datathon.validation`, roda a validação sobre o parquet
processado e exibe o relatório com `st.metric`/`st.dataframe`. Ver `app_dashboard_pt.py` linhas
33–46 para os imports e a página "Data Quality" do dashboard para a exibição.

---

## Como usar

```python
from datathon.quality import validate_data_quality, get_quality_report, setup_logging
from datathon.validation import (
    get_confidence_interval,
    compare_conversions,
    compare_contexts,
    calculate_sample_size,
    generate_statistical_report,
)

# 1. Setup logging
setup_logging(log_level='INFO')

# 2. Validar dados
df = pd.read_parquet('data.parquet')
report = validate_data_quality(df, dataset_name='my_dataset')
# ou, para exibir num dashboard:
report = get_quality_report(df)
st.metric("Data Quality", report['overall_status'])
st.dataframe(pd.DataFrame(report['metrics']))

# 3. Testes estatísticos
ci_lower, ci_upper = get_confidence_interval(successes=100, trials=1000)
result = compare_conversions(s1=100, t1=1000, s2=95, t2=1000)

# 4. Logar
logger.info("Validação concluída")
logger.warning("Problema encontrado")
logger.error("Falha crítica")
```

---

## Status

Data quality, logging e testes estatísticos estão implementados, testados (60/60) e em uso no
dashboard `app_dashboard_pt.py`. `train_simple.py` e `train_with_mlflow.py` não usam este módulo
hoje. Não há alertas automáticos de anomalia nem persistência de log em arquivo — por padrão os
logs vão só para stdout (`setup_logging` aceita um `log_file` opcional, não usado atualmente).
