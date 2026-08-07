# Datathon 7MLET - POS TECH

## DESAFIO

O Datathon propõe um desafio único no domínio financeiro regulado: projetar uma plataforma de experimentação adaptativa para ofertas, mensagens ou próximos passos em canais digitais. Cada grupo constrói uma solução end-to-end de Machine Learning Engineering e demonstra como ela seria operada com segurança, observabilidade, avaliação e governança.

O objetivo não é reproduzir um sistema bancário real, mas sim mostrar maturidade técnica: formular o problema, construir baselines, versionar dados, servir componentes, avaliar qualidade, monitorar risco, documentar limitações e explicar decisões para públicos técnicos e de negócio, considerando o seguinte caso:

### Caso de Uso

Uma instituição financeira digital precisa decidir, em diferentes canais, qual oferta, mensagem ou próximo passo apresentar para cada cliente elegível. Regras fixas e testes A/B longos desperdiçam tráfego, demoram para reagir a mudanças de contexto e dificultam a personalização responsável. Esse é o ponto central de uma abordagem de **multi-armed bandit**: identificar comportamentos distintos, equilibrar exploração e explotação e aprender com respostas observadas sem congelar a decisão em regras estáticas. A solução deve incluir um assistente com LLM que resuma experimentos, recupere políticas internas sintéticas e explique decisões.

### Referências Algorítmicas

| Algoritmo | Papel no desafio | Evidência esperada |
|-----------|------------------|-------------------|
| **Thompson Sampling** | Exploração bayesiana sob incerteza para modelar conversão, clique ou recompensa esperada por braço. | Priors documentados, comparação com baseline e análise de exploração. |
| **Nilos-UCB** | Família UCB para selecionar ações com base em recompensa esperada e incerteza. | Fórmula, implementação ou adaptação justificada, com análise do trade-off entre confiança, exploração e conversão. |
| **Baseline determinístico** | Política simples de controle (regra fixa, melhor braço histórico ou segmentação inicial). | Métrica comparativa clara para mostrar ganho ou limitação da política adaptativa. |

O grupo pode implementar Thompson Sampling, Nilos-UCB, LinUCB ou outra variação contextual, desde que explique a escolha, mostre como o contexto entra na decisão e documente o tratamento de recompensas atrasadas.

## Dados, Regras e Bases Kaggle

Use uma base Kaggle compatível com marketing, ofertas, propensão, campanhas, recomendação ou conversão como referência factual e crie uma camada sintética de experimentação adaptativa sobre ela (impressões, ações, contexto, recompensas, eventos atrasados, documentos de política comercial e suitability para RAG). 

**Não use dados reais de clientes, identificadores, patrimônio, renda, gênero, raça ou regras comerciais privadas. Mantenha decisões sensíveis com humano no loop e documente base legal, finalidade, minimização e retenção.**

### Bases Kaggle Recomendadas

| Base Kaggle | Como usar no desafio |
|-------------|----------------------|
| **bank-marketing** (henriqueyamahata) | Campanhas bancárias, propensão de conversão e decisão de oferta. |
| **bank-marketing-data-set** (tunguz) | Variação do problema de marketing bancário para comparação. |
| **bank-term-deposit-subscription** (dharmik34) | Assinatura de depósito a prazo como proxy de conversão. |
| **telemarketing-jyb-dataset** (aguado) | Campanhas de contato e resposta, útil para comparação de canal ou abordagem. |

Outras bases são aceitas se o grupo justificar a aderência e documentar fonte, versão, licença, colunas, target e limitações. Descarte colunas de vazamento temporal (ex.: duration no Bank Marketing) e preserve a referência ao Kaggle.

### Entrega Mínima de Dados

Cada grupo deve entregar, no mínimo, uma camada derivada versionada contendo:

- `data/kaggle/README.md` (fonte, link, versão, licença);
- `data/processed/` (base tratada sem vazamento);
- `data/synthetic_enrichment/` com offer_catalog, offer_events e delayed_rewards;
- `data/golden_set/evaluation_cases.jsonl` com pelo menos 20 casos;
- `reports/data-generation.md` documentando processo, sementes, hipóteses, limitações e riscos.

## Entregáveis Obrigatórios

Os entregáveis abaixo são organizados em **nove etapas acumulativas (0–8)**. Cada etapa traz um objetivo, uma lista detalhada de artefatos técnicos exigidos e o critério de evidência de aceite que a banca usa para considerar a etapa cumprida.

**Uma etapa posterior não compensa uma etapa anterior ausente**: um grupo com modelo sofisticado, mas sem dataset rastreável, avaliação reproduzível ou aprovação de ciclo de vida é penalizado na validação técnica. A ausência de um subitem dentro de uma etapa não é compensada pela presença dos demais.

### Etapa 0 — Organização do Projeto

**Objetivo**: preparar um repositório público reutilizável por outra pessoa sem contexto oral do grupo.

#### Artefatos Exigidos

- URL pública do repositório com nome no padrão `datathon-7mlet-grupo-XX`.
- `README.md` com visão do problema, escopo, escolhas de design, instruções de execução local, mapa de pastas, lista de comandos e limitações.
- `pyproject.toml` declarando dependências, versão de Python, ponto de entrada e ferramentas de desenvolvimento.
- `.env.example` listando variáveis de ambiente necessárias, sem valores reais.
- Licença, `.gitignore` adequado e ausência de segredos, dados sensíveis ou modelos binários grandes versionados.
- Histórico de commits que mostre evolução do trabalho, não apenas um commit final.

#### Evidência de Aceite

Uma pessoa externa consegue instalar dependências, entender o fluxo e executar pelo menos um comando de validação sem precisar de explicação oral do grupo.

---

### Etapa 1 — Base Kaggle e EDA

**Objetivo**: transformar uma base Kaggle compatível em fonte confiável para experimentação.

#### Artefatos Exigidos

- `data/kaggle/README.md` com link, versão, fonte, licença, limitações e instruções de download.
- Dicionário de dados, notebook de EDA e relatório de qualidade.
- Camada de dados em código que carrega a base, registra fonte/versão/licença e gera os datasets derivados documentados.
- Decisão documentada sobre colunas que geram vazamento temporal ou pós-contato, com justificativa de descarte ou tratamento.

#### Evidência de Aceite

A banca consegue rastrear a origem do dataset, entender as variáveis usadas e verificar que não houve vazamento de informação pós-contato no modelo de decisão.

---

### Etapa 2 — Enriquecimento Sintético

**Objetivo**: criar a camada de experimentação adaptativa sobre o dataset escolhido.

#### Artefatos Exigidos

- Catálogo sintético de braços/ofertas separado fisicamente da base Kaggle original.
- Eventos de impressão, contexto de decisão e recompensas intermediárias com sementes aleatórias controladas.
- Modelagem de delayed rewards e do horizonte temporal documentada.
- Schema dos arquivos sintéticos e processo de geração descritos no repositório.

#### Evidência de Aceite

Os arquivos sintéticos têm schema documentado e explicam como braços, contexto, recompensa e horizonte temporal foram definidos, separados da base Kaggle original.

---

### Etapa 3 — Baseline e Estratégia Algorítmica

**Objetivo**: comparar política simples com abordagem multi-armed bandit.

#### Artefatos Exigidos

- Pelo menos um baseline determinístico implementado.
- Implementação ou simulação de Thompson Sampling.
- Referência experimental ou justificativa explícita de Nilos-UCB na análise algorítmica.
- Métricas de recompensa, regret, exploração e conversão simulada calculadas e reportadas.
- Tratamento de cold-start e de recompensas atrasadas descrito.

#### Evidência de Aceite

O grupo apresenta comparação quantitativa entre baseline e política adaptativa, com justificativa do algoritmo escolhido e do tratamento de cold-start e delayed rewards.

---

### Etapa 4 — Avaliação Offline e Golden Set

**Objetivo**: medir qualidade técnica e risco antes de servir a decisão.

#### Artefatos Exigidos

- Script ou notebook de avaliação offline reproduzível por linha de comando ou notebook, com métricas justificadas.
- Golden set com no mínimo 20 exemplos versionados em `data/golden_set/evaluation_cases.jsonl` ou formato equivalente.
- Cobertura do golden set sobre casos típicos, casos de borda, segmentos sintéticos elegíveis e cenários adversariais.
- Cada caso traz contexto, ação esperada, recompensa esperada, justificativa e critério explícito de pass/fail.
- Matriz de métricas, análise de sensibilidade e análise de fairness de exposição entre segmentos sintéticos.

#### Evidência de Aceite

As métricas são reproduzíveis, o golden set está versionado e a análise explica limitações, vieses e condições em que a política não deve ser usada.

---

### Etapa 5 — Serviço ou Interface Demonstrável

**Objetivo**: expor a decisão de forma controlada e auditável.

#### Artefatos Exigidos

- API, CLI, notebook executável ou app demonstrável que recebe um contexto e devolve uma decisão.
- Contrato de entrada e saída documentado, com exemplo de chamada e tratamento de erro.
- Log auditável de decisão com reason codes, braço selecionado e versão da política aplicada.
- Comando único ou script que permita reproduzir o pipeline ponta a ponta em ambiente local.
- Suíte mínima de testes automatizados cobrindo contratos de dados, política e registro de decisão.

#### Evidência de Aceite

A banca consegue executar uma decisão de exemplo, ver o braço selecionado, a justificativa, a versão da política e o registro auditável gerado.

---

### Etapa 6 — Arquitetura-Alvo Azure

**Objetivo**: demonstrar como a solução seria operada em Azure.

#### Artefatos Exigidos

- `docs/architecture-azure.md` com diagrama Mermaid e mapeamento de serviços Azure.
- Plano de deploy e estimativa qualitativa de custo.
- Cobertura das camadas de compute, API, dados, IA/RAG, observabilidade, segurança, identidade e governança.
- Plano de gestão de segredos e credenciais usando Azure Key Vault e Managed Identity.

#### Evidência de Aceite

A arquitetura usa exclusivamente Azure, cobre as camadas acima e justifica trade-offs sem depender de outro provedor de nuvem.

---

### Etapa 7 — Ciclo de Vida MLOps

**Objetivo**: mostrar como novas políticas seriam testadas, aprovadas e promovidas.

#### Artefatos Exigidos

- Plano de retreino com critérios de promoção, approval gate, rollback e versionamento de política.
- Monitoramento de drift e de recompensa documentado.
- Rastreio de experimentos em MLflow ou ferramenta equivalente.
- Procedimento de teste, aprovação humana estruturada e promoção de novas políticas para produção controlada.

#### Evidência de Aceite

O grupo demonstra como uma nova hipótese de oferta/canal/mensagem sairia de experimento para produção controlada, com aprovação humana e rollback documentado.

---

### Etapa 8 — Governança, Demo Day e Relatórios

**Objetivo**: fechar a entrega com responsabilidade operacional e narrativa coerente.

#### Artefatos Exigidos

- `docs/model-card.md` com nome do modelo, versão, dados de treino e avaliação, métricas, intended use, out-of-scope use, análise de fairness, vieses conhecidos e limitações técnicas.

- `docs/system-card.md` com escopo, fluxo de decisão, dependências, guardrails, cenários de risco (reward hacking, manipulação do contexto, abuso do assistente, violação de suitability) e plano de monitoramento.

- `docs/lgpd-plan.md` com base legal, finalidade, minimização, ciclo de retenção, mapeamento de identificadores e atributos protegidos, política de logs/telemetria e plano de resposta a incidentes.

- **Relatório técnico** de até 10 páginas cobrindo:
  - Problema
  - Base escolhida
  - Enriquecimento sintético
  - Modelagem como multi-armed bandit
  - Comparação quantitativa
  - Arquitetura-alvo Azure
  - Ciclo MLOps
  - Limitações
  - Riscos
  - Hipóteses
  - Trabalhos futuros
  - Referências

- **Pitch** de até 10 minutos seguido por 5 minutos de perguntas, com:
  - Slides versionados em PDF ou formato aberto
  - Roteiro cobrindo problema/abordagem/demonstração/evidências/riscos/governança/impacto

- **Demonstração** ao vivo ou gravada da plataforma é desejável e soma pontos extras:
  - Grupo deve indicar o cenário
  - Registrar plano de contingência
  - Versionar a gravação ou dataset de demonstração

- **Cobertura no pitch** dos critérios de apresentação:
  - **FinOps**: ROI, custo por serviço Azure, TCO
  - **Justificativa de arquitetura técnica** com diagrama, fronteiras e alternativas descartadas
  - **Cenários de escala e redução** por volume de requisições

- Plano de revisão periódica do model card e do system card com responsáveis e cadência definidos.

#### Evidência de Aceite

A banca encontra narrativa coerente de problema, solução, evidências, riscos, governança e valor de negócio, sem alegar prontidão para produção real regulada.

---

## Critérios de Avaliação

A avaliação segue o contrato da Fase 05:

| Dimensão | Peso | O que a banca procura |
|----------|------|----------------------|
| **Critérios de negócio** | 30% | Aderência ao problema escolhido, clareza de impacto, viabilidade, comunicação executiva |
| **Validação técnica global** | 70% | Pipeline, MLOps, avaliação, observabilidade, segurança, governança, documentação e uso de PyTorch/MLflow quando aplicável |

Os grupos devem definir métricas específicas para o desafio. Essas métricas precisam ser justificadas no relatório técnico e conectadas ao impacto de negócio, mas não substituem os critérios oficiais da fase.

---

## Checklist antes do Demo Day

- ☐ README explica desafio, execução local e limitações; pipeline usa base Kaggle compatível com fonte, versão, licença e limitações.

- ☐ Base processada e enriquecimento sintético documentados e separados da base Kaggle original; experimentos rastreados em MLflow.

- ☐ Baseline e abordagem principal comparados com métricas justificadas; análise referencia Thompson Sampling e Nilos-UCB.

- ☐ Avaliação inclui golden set com pelo menos 20 exemplos; guardrails testados com cenários adversariais.

- ☐ Camada de retreino, teste, aprovação e promoção de novas políticas documentada.

- ☐ Serviço, API, notebook executável ou interface demonstrável funciona com instruções claras e log auditável de decisão.

- ☐ Arquitetura-alvo e plano de deploy usam exclusivamente serviços Azure, com plano de segredos via Key Vault e Managed Identity.

- ☐ Model Card, System Card e plano LGPD completos.

- ☐ Pitch separa problema, abordagem, demonstração, evidências, riscos e impacto.

- ☐ Pitch cobre FinOps (ROI, custo qualitativo por serviço Azure, TCO).

- ☐ Pitch justifica arquitetura técnica com diagrama, fronteiras e alternativas descartadas e apresenta cenários de escala e redução.

- ☐ Pitch inclui demonstração ao vivo ou gravada da plataforma em operação, com plano de contingência (desejável; soma pontos extras).
