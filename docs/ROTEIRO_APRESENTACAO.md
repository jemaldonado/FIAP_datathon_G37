# Roteiro do Vídeo Pitch (até 6 minutos)


## Checklist oficial do demo day — onde cada item é coberto

Checklist copiado de `descricao-projeto` (seção "Checklist antes do demo day"), com o
comprovante de cada item já entregue e o momento do vídeo (se houver) que o demonstra:

| Item do edital | Status | Evidência / onde mostrar |
|---|---|---|
| Repositório organizado, `requirements.txt`/`pyproject.toml` | pronto | `pyproject.toml`, `requirements.txt` — não precisa aparecer no vídeo |
| Notebook de EDA com base Kaggle limpa e referenciada | pronto | `notebooks/` — citar de passagem em 0:30–1:10 |
| Baseline e Modelo Adaptativo implementados e comparados | pronto | `notebooks/datathon_main.ipynb` — baseline 11,27% vs Thompson 14,97% (+3,70 p.p.), reproduzível; falar em 1:10–1:45 |
| README/Notebook com 5 casos de teste (Golden Set) | pronto | `data/golden_set/golden_set.json`, seção README "Golden set — 5 perfis validados"; citar em 3:45–4:15 |
| Código executável que retorna a predição | pronto | `POST /recommend` via Swagger, demo ao vivo em 1:45–2:45 |
| README com link da base + parágrafo de infraestrutura cloud | pronto | seção README "Arquitetura" + `docs/ARQUITETURA.md`; citar em 4:50–5:25 |
| Tracking de experimentos via MLOps (MLflow) | pronto | `.mlflow/mlflow.db` + `app_mlflow_showcase.py`, com tela dedicada em 4:15–4:50 |
| Vídeo de até 6 min, código funcionando, escolhas justificadas | pendente (este roteiro) | gravar seguindo o roteiro abaixo |

**Além do mínimo do edital:** o projeto implementa canary deployment de verdade (não só
teórico) e documenta a arquitetura AWS/Azure em nível profissional com custos reais —
não é exigido para nota, mas é a parte mais forte para mostrar maturidade técnica (peso de
30% do critério é justamente "clareza do problema e impacto da solução").

## Antes de gravar

Deixe tudo já treinado e rodando para não perder tempo de gravação com setup:

```bash
python scripts/train_with_mlflow.py         # garante thompson_model.json atualizado + run nova no MLflow
pytest tests/ -v                             # confirma 60/60 passando (mostrar rapidamente é opcional)
python src/datathon/api/app.py               # Terminal 1 — deixa a API no ar
streamlit run app_canary_arm_swap_demo.py    # Terminal 2 — demo do caso real de canary
streamlit run app_mlflow_showcase.py         # Terminal 3 (opcional) — evidência do MLflow
streamlit run app_dashboard_pt.py            # Terminal 4 (opcional) — página "Thompson Aprende?"
streamlit run app_dashboard.py               # Terminal 5 (opcional) — visão geral do pipeline
```

Abra com antecedência, em abas já prontas: `http://localhost:5000/apidocs` (Swagger) e a
página do Streamlit `app_canary_arm_swap_demo.py` (porta padrão 8501; os demais dashboards
sobem em portas seguintes automaticamente se abertos ao mesmo tempo). Tenha 2–3 perfis de
cliente já digitados num arquivo de texto para colar no Swagger sem digitar ao vivo (evita
erro de digitação na gravação). No app de canary, o cliente de exemplo (idade 28, "admin") já
vem fixo — não precisa preencher nada.

**Dashboards do projeto** (instalação, propósito e comando de cada um):
ver seção [Dashboards](../README.md#dashboards) do `README.md`. Os dois usados no roteiro
abaixo são `app_canary_arm_swap_demo.py` (2:45–3:45) e `app_mlflow_showcase.py` (4:15–4:50);
`app_dashboard.py` e `app_dashboard_pt.py` são material de apoio para perguntas da banca fora
do corte de 6 minutos — a página "Thompson Aprende?" de `app_dashboard_pt.py` é o lugar certo
pra aprofundar a seção 1:10–1:45 se sobrar tempo ou em Q&A.

## Roteiro (cronometrado — soma 6:00)

### 0:00–0:30 — O problema de negócio

> "Um banco digital decide todo dia qual oferta apresentar para cada cliente elegível — cartão,
> depósito, próximo contato. Regra fixa desperdiça tráfego em quem já não converteria; testar
> tudo por A/B tradicional é lento para reagir. Este projeto responde: como decidir, cliente a
> cliente, com um algoritmo que aprende sozinho qual abordagem funciona melhor para cada
> perfil — sem regra fixa, sem esperar um teste A/B terminar?"

### 0:30–1:10 — Dados e abordagem (tela: README ou notebook de EDA)

> "Usamos a base pública `bank-marketing` do Kaggle/UCI — 41 mil contatos de telemarketing de um
> banco português, alvo é se o cliente aceitou um depósito a prazo. Removemos a coluna
> `duration` porque só é conhecida depois da ligação — usá-la seria vazamento de dados.
>
> Como baseline, comparamos com a estratégia mais simples: manter o mix histórico real de
> canais, sem contexto — 11,27% de conversão. O Thompson Sampling, aprendendo a taxa real de
> cada canal, converge para o canal melhor rápido e chega a 14,97% — quase 4 pontos percentuais
> a mais, gastando menos de 1% das tentativas explorando o canal pior. Contra esse baseline,
> implementamos Thompson Sampling **contextual**: 12 modelos, um para cada combinação de faixa
> etária (Young, Prime, Mature, Senior) e categoria profissional (Technical, Business, Other)."

### 1:10–1:45 — Como o modelo decide (tela: diagrama simples ou código do bandit)

> "Cada contexto mantém uma distribuição Beta por estratégia de contato. Para cada cliente
> novo, o modelo amostra uma probabilidade de sucesso por estratégia, escolhe a de maior valor
> amostrado, observa se converteu, e atualiza. A conversão varia de **8,7% no pior contexto até
> 42,2% no melhor** — mais de 30 pontos percentuais que uma política única, sem contexto,
> jamais capturaria."

### 1:45–2:45 — Demo ao vivo: a API respondendo (tela: Swagger)

1. `POST /recommend` com 2 perfis diferentes já preparados (ex.: jovem técnico vs. aposentado
   sênior), mostrando a resposta JSON com `recommended_arm`, `expected_conversion` e
   `rationale`.
   > "A API recebe idade, profissão, estado civil etc., identifica o contexto do cliente e
   > devolve a recomendação com a taxa de conversão esperada e a justificativa."

### 2:45–3:45 — Canary deploy: um caso real de retreino mudando a recomendação (tela: app_canary_arm_swap_demo.py)

> "Modelos são retreinados com dados novos — e isso pode mudar a recomendação. O dataset é
> ordenado pela data real de contato, então comparamos um snapshot treinado só com os primeiros
> 70% dos contatos com o modelo final, treinado com a campanha completa. Para clientes jovens
> em cargos técnicos, o snapshot antigo recomenda campanha por e-mail, com 10,72% de conversão
> esperada. Com mais dados reais — incluindo um período de conversão bem mais alta — o modelo
> passa a recomendar contato celular padrão, com 19,66%: quase 1,8x maior, e troca a oferta
> vencedora. Uma troca real, não simulada artificialmente."

Mostrar a tela do Streamlit: já abre com a comparação estática (baseline vs. canary). Clicar
**▶️ Iniciar Canary** e depois **🔄 Rodar Simulação** para revelar o gráfico ao vivo.

> "É exatamente esse tipo de mudança que o canary deploy existe para validar: expomos a
> nova versão a uma fração do tráfego, monitoramos se ela realmente converte melhor, e só
> promovemos para 100% dos clientes se os números confirmarem — em vez de aplicar o retreino
> cegamente."

### 3:45–4:15 — Resultados e validação (tela: golden set no README)

> "Validamos com um golden set de 5 perfis representativos, conferindo se a recomendação faz
> sentido para cada um, com 60 testes automatizados cobrindo API, bandit e canary deploy —
> todos passam."

### 4:15–4:50 — MLflow: tracking de experimentos (tela: app_mlflow_showcase.py, porta 8502)

> "Cada treino fica registrado no MLflow — não é só um número solto no README. Aqui estão as
> runs reais, direto do tracking store: parâmetros do modelo e as métricas de baseline vs.
> Thompson que mostrei antes, 11,27% para 14,97%, rastreáveis run por run."

Mostrar o gráfico baseline vs. Thompson do dashboard, puxado direto de `.mlflow/mlflow.db` —
não é a UI do MLflow, mas os dados são os mesmos, sem precisar abrir outra ferramenta.

### 4:50–5:25 — Nuvem (tela: `docs/ARQUITETURA.md` ou `docs/architecture/`)

> "Para produção, a arquitetura é enxuta de propósito: dados e modelo (poucos KBs) em object
> storage — S3 na AWS ou Blob Storage na Azure —, treino como função serverless, API e
> dashboard como serviços PaaS gerenciados. Sem clusters de treino pesados, porque o problema
> não exige isso. O canary deploy que acabamos de mostrar também tem equivalente de
> infraestrutura em cada nuvem — detalhado em `docs/architecture/`."

### 5:25–6:00 — Fechamento

> "Em resumo: um bandit contextual que aprende com cada cliente, valida contra baseline
> (11,27% para 14,97%), versiona cada experimento no MLflow e usa canary deploy para validar
> retreinos com segurança antes de impactar todos os clientes. Obrigado!"


