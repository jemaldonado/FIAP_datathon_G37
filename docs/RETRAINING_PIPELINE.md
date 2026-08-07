# Model Retraining Pipeline

Documentação do pipeline de retrenamento, avaliação e promoção de modelos Thompson Sampling.

## Visão Geral

O pipeline automatiza o processo de:

1. **Retrenamento** - Treina novo modelo com dados atualizados
2. **Avaliação** - Valida qualidade comparando com modelo em produção
3. **Promoção** - Se aprovado, promove novo modelo para produção

## Arquitetura

```
┌─────────────────────────────────────────┐
│ Dados Novos/Atualizados                 │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 1. retrain_model.py                     │
│   - Treina novo modelo                  │
│   - Versionado: thompson_model_v....json│
│   - Loga em MLflow                      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 2. evaluate_model.py                    │
│   - Compara vs produção                 │
│   - Valida Golden Set                   │
│   - Detecta regressões                  │
└────────────┬────────────────────────────┘
             │
         ┌───┴────┐
         │         │
       Falha    Sucesso
         │         │
         ▼         ▼
      Rejeita  ┌────────────────────┐
              │ 3. Promoção        │
              │ - Copy para prod    │
              │ - thompson_model.json
              └────────────────────┘
```

## Scripts

### 1. retrain_model.py

Treina novo modelo com dados atualizados.

**Uso:**
```bash
# Treinar com todos os dados (padrão)
python scripts/retrain_model.py

# Treinar com 80% dos dados (simular novos dados chegando)
python scripts/retrain_model.py --sample-ratio 0.8

# Treinar com versão customizada
python scripts/retrain_model.py --version v_custom_2026_07_07

# Treinar sem MLflow (rápido para testes)
python scripts/retrain_model.py --no-mlflow

# Treinar em AWS
python scripts/retrain_model.py --env aws
```

**Saída:**
```
[INFO] RETRAINING THOMPSON SAMPLING MODEL - v20260707_123456
[INFO] Loading data from data/processed/bank_marketing_primary.parquet
[INFO] Loaded 41,188 customers
[INFO] Training on 41,188 customers...
[INFO]   Processed 10,000 customers
[INFO] ...
[INFO] Model saved: thompson_model_v20260707_123456.json (4.7 KB)
[INFO] Overall conversion: 11.27%
[INFO] Best context: 41.11%
[INFO] Worst context: 7.50%
[INFO] Spread: 33.6 pp
```

**Arquivos criados:**
- `data/models/thompson_model_v{timestamp}.json` - Versão do modelo
- `.mlflow/mlflow.db` - Métricas em MLflow

---

### 2. evaluate_model.py

Valida qualidade do novo modelo contra o em produção.

**Uso:**
```bash
# Avaliar modelo contra produção
python scripts/evaluate_model.py --new-model data/models/thompson_model_v20260707_123456.json

# Com modelo antigo explícito
python scripts/evaluate_model.py \
  --new-model data/models/thompson_model_v20260707_123456.json \
  --old-model data/models/thompson_model_v20260707_100000.json

# Salvar relatório
python scripts/evaluate_model.py \
  --new-model data/models/thompson_model_v20260707_123456.json \
  --save-report outputs/eval_report_latest.json
```

**Saída:**
```
[INFO] MODEL EVALUATION
[INFO] Loading model from data/models/thompson_model_v20260707_123456.json
[INFO] Comparing against data/models/thompson_model.json

[INFO] Context Performance:
  Mature_Business: 8.7% (180/2,079)
  Mature_Other: 8.4% (384/4,575)
  Mature_Technical: 10.1% (514/5,087)
  Prime_Business: 10.5% (331/3,157)
  ...

[INFO] Regression Analysis:
  No regressions detected ✓

[INFO] APPROVAL: ✓ Model approved for production
```

**Critérios de aprovação:**
- Nenhuma regressão detectada (threshold: -2% por contexto)
- Golden Set validado (contextos corretos)
- Ao menos 1 contexto sem melhora (estável)

**Saída (JSON):**
```json
{
  "timestamp": "v20260707_123456",
  "new_model": "data/models/thompson_model_v20260707_123456.json",
  "new_metrics": {
    "Mature_Business": {
      "context": ["Mature", "Business"],
      "trials": 2079,
      "successes": 180,
      "rate": 0.087
    },
    ...
  },
  "comparison": {
    "comparison": {
      "Mature_Business": {
        "new_rate": 0.087,
        "old_rate": 0.087,
        "diff": 0.0,
        "pct_change": 0.0,
        "regression": false
      },
      ...
    },
    "regressions": [],
    "has_regressions": false
  },
  "approved": true
}
```

---

### 3. compare_models.py

Compara dois modelos lado a lado (útil para análise visual).

**Uso:**
```bash
# Comparar modelo novo vs produção
python scripts/compare_models.py \
  --model1 data/models/thompson_model.json \
  --model2 data/models/thompson_model_v20260707_123456.json

# Salvar relatório
python scripts/compare_models.py \
  --model1 data/models/thompson_model.json \
  --model2 data/models/thompson_model_v20260707_123456.json \
  --save-report outputs/comparison_latest.json
```

**Saída:**
```
==========================================================================================
MODEL COMPARISON REPORT
==========================================================================================
Model 1: thompson_model.json
Model 2: thompson_model_v20260707_123456.json
==========================================================================================

Context              Model 1      Model 2      Diff       Change %   Status
------------------------------------------------------------------------------------------
Mature_Business       8.70%       8.70%       +0.00%       +0.0%   → SAME
Mature_Other          8.40%       8.40%       +0.00%       +0.0%   → SAME
Mature_Technical     10.10%      10.10%       +0.00%       +0.0%   → SAME
Prime_Business       10.50%      10.50%       +0.00%       +0.0%   → SAME
...
Senior_Other         41.10%      41.10%       +0.00%       +0.0%   → SAME

==========================================================================================
SUMMARY
==========================================================================================
Total contexts: 12
Improvements:   0
Regressions:    0
No change:      12
Average diff:   +0.00%
==========================================================================================

✓ No regressions detected
```

---

### 4. pipeline_retrain_eval.py

Orquestra o pipeline completo: treina → avalia → promove.

**Uso:**
```bash
# Pipeline completo com todos os dados
python scripts/pipeline_retrain_eval.py

# Pipeline com 80% dos dados (simular novos dados)
python scripts/pipeline_retrain_eval.py --sample-ratio 0.8

# Pipeline em AWS
python scripts/pipeline_retrain_eval.py --env aws
```

**Fluxo:**
1. Treina novo modelo (mesmo que retrain_model.py)
2. Avalia novo modelo (mesmo que evaluate_model.py)
3. Se aprovado:
   - Promove para `data/models/thompson_model.json`
   - Salva log do pipeline em `outputs/retraining/pipeline_*.json`
4. Se rejeitado:
   - Para o processo
   - Modelo versionado fica disponível mas não é promovido

**Saída:**
```
[INFO] RETRAIN → EVALUATE → PROMOTE PIPELINE
[INFO] [STEP 1/3] Training new model...
[INFO] Running: python scripts/retrain_model.py --sample-ratio 1.0 --version v20260707_123456
[INFO] TRAINING COMPLETE!
[INFO] ✓ Training successful

[INFO] [STEP 2/3] Evaluating model...
[INFO] Running: python scripts/evaluate_model.py --new-model data/models/thompson_model_v20260707_123456.json ...
[INFO] APPROVAL: ✓ Model approved for production
[INFO] ✓ Evaluation successful - Model approved

[INFO] [STEP 3/3] Promoting to production...
[INFO] ✓ Model promoted to production: data/models/thompson_model.json
[INFO] PIPELINE COMPLETE - SUCCESS
```

**Log salvo (outputs/retraining/pipeline_*.json):**
```json
{
  "timestamp": "20260707_123456",
  "status": "success",
  "steps": {
    "train": {
      "status": "success",
      "model_path": "data/models/thompson_model_v20260707_123456.json"
    },
    "evaluate": {
      "status": "approved",
      "report": { ... }
    },
    "promote": {
      "status": "success",
      "production_path": "data/models/thompson_model.json"
    }
  }
}
```

---

## Versionamento de Modelos

Os modelos são versionados com timestamp: `thompson_model_v{YYYYMMDD}_{HHMMSS}.json`

```
models/
├── thompson_model.json                 (produção - última versão aprovada)
├── thompson_model_v20260707_120000.json (aprovado semana passada)
├── thompson_model_v20260707_123456.json (aprovado hoje - V1)
├── thompson_model_v20260707_124500.json (recusado hoje - regressão detectada)
└── thompson_model_v20260707_125000.json (aprovado hoje - V2)
```

**Política:**
- `thompson_model.json` → Sempre a versão em produção (última aprovada)
- Versões antigas → Mantidas para auditoria e rollback
- Cleanup → Remover versões > 30 dias (opcional)

---

## MLflow Tracking

Cada retrenamento é rastreado em MLflow:

```bash
# Ver dashboard MLflow
mlflow ui --backend-store-uri sqlite:///$(pwd)/.mlflow/mlflow.db
```

**Métricas logadas por treino:**
- `overall_conversion` - Taxa geral de conversão
- `best_context_conversion` - Melhor contexto
- `worst_context_conversion` - Pior contexto
- `spread_pp` - Diferença (em pontos percentuais)
- `contexts_with_data` - Contextos com dados
- `conversion_{age_group}_{job_category}` - Taxa por contexto (12 métricas)

**Parâmetros logados:**
- `version` - ID da versão
- `sample_ratio` - Fração de dados usada
- `n_customers` - Número de clientes treinados

---

## Fluxo Recomendado

### Desenvolvimento Local
1. Treinar novo modelo: `python scripts/retrain_model.py --sample-ratio 0.8`
2. Comparar visualmente: `python scripts/compare_models.py --model1 ... --model2 ...`
3. Avaliar qualidade: `python scripts/evaluate_model.py --new-model ... --save-report ...`
4. Se OK, pipeline completo: `python scripts/pipeline_retrain_eval.py --sample-ratio 0.8`
