# Plano LGPD — Datathon FIAP G37 (datathon-clean)

> Postura de proteção de dados sob a Lei Geral de Proteção de Dados (LGPD).  
> Esta plataforma **não processa dados pessoais reais de clientes**.  
> Usa base pública anônima (Kaggle/UCI). Os 4 braços de campanha do modelo em produção são
> segmentos reais (canal × primeira/repetida tentativa de contato) relabeled com nomes de
> negócio — conversões 100% reais, não uma camada de dado sintético (ver
> `datathon.bandit.assign_arm`). Existe, separadamente, um exercício exploratório
> (`campaign_synthesis.py`) que simula taxas hipotéticas — não usado no pipeline de produção.  
> Este plano documenta essa fronteira e descreve os controles que se aplicariam se o desenho fosse operado com dados reais — a rubrica do Datathon avalia a **postura**, não só a ausência de dados.

**Projeto:** Thompson Sampling contextual (API Flask)  
**Etapa do desafio:** Etapa 8 — Governança (`plan/Datathon-7MLET.md`)

---

## 1. Escopo e honestidade

| Afirmação | Status |
|---|---|
| Dados reais de clientes do banco | **Não usados** |
| Identificadores (nome, CPF, telefone, e-mail) | **Ausentes** em todos os schemas |
| Atributos protegidos (gênero, raça, renda, patrimônio) | **Ausentes** / não usados na decisão |
| Dataset de referência | Kaggle `henriqueyamahata/bank-marketing` (já anonimizado) |
| Campanhas Email / SMS / Call_Premium | **Nomes relabeled** de segmentos reais (canal × campaign); conversões 100% reais (ver `datathon.bandit.assign_arm`) |
| Sistema de produção regulado | **Não** — entrega acadêmica do Datathon |

---

## 2. Mapeamento de dados

| Camada | Conteúdo | Dado pessoal? |
|---|---|---|
| `data/kaggle/` | CSVs públicos de campanha bancária (UCI/Kaggle); sem nomes ou IDs | **Não** — pesquisa pública anonimizada; referência factual |
| `data/processed/` | Parquet limpo; coluna `duration` removida no ETL | **Não** |
| Enrichment exploratório, não usado em produção (`campaign_synthesis.py`) | Taxas hipotéticas de Email/SMS/Premium; eventos simulados — script órfão, nunca executado no pipeline real | **Não** — sintético por construção; não é a fonte dos braços do modelo servido pela API |
| `data/models/thompson_model.json` | Posteriors Beta por contexto (age_group × job_category); os 4 braços são segmentos reais relabeled (ver `datathon.bandit.assign_arm`), não estimativas | **Não** — agregados estatísticos |
| `data/golden_set/golden_set.json` | 5 perfis fictícios para validação | **Não** — casos de teste |
| Request `POST /recommend` | `age`, `job`, `marital`, `education`, `contact`, `campaign` | Em produção seria contexto de cliente; **aqui** é input de demo / perfil sintético |
| Response `/recommend` | `recommended_arm`, `arm_name`, `expected_conversion`, `context`, `rationale` | **Não** — não grava identidade |
| MLflow (`.mlflow/`) | Métricas de experimento | **Não** — agregados |

### Identificadores

| Identificador | Presente? |
|---|---|
| Nome | Não |
| CPF / documento | Não |
| Conta bancária | Não |
| Telefone / e-mail | Não |
| Device ID / IP | Não |
| ID de decisão | Não implementado ainda (recomendado: UUID no futuro log de auditoria) |

### Atributos sensíveis / protegidos

| Atributo | Presente? | Uso na decisão |
|---|---|---|
| Gênero | Não | — |
| Raça / etnia | Não | — |
| Renda / patrimônio | Não | — |
| Saúde / biometria | Não | — |
| Idade (número) | Sim (contexto) | Agregada em `age_group` (Young/Prime/Mature/Senior) |
| Profissão | Sim (contexto) | Agregada em `job_category` (Technical/Business/Other) |

A decisão do bandit usa apenas o **contexto agregado** (`age_group` × `job_category`), não o perfil bruto completo.

---

## 3. Finalidade e base legal (desenho para operação real)

### Finalidade (única e específica)

Decidir **qual estratégia de campanha** (Cellular_Standard, Email_Campaign, SMS_Alert, Call_Premium) apresentar a um perfil elegível, e permitir auditoria dessa recomendação.  
**Sem uso secundário** (não vender dados, não perfilar crédito, não precificar).

### Base legal (se operasse com clientes reais)

| Base | Artigo LGPD | Aplicação |
|---|---|---|
| Interesse legítimo | art. 7º, IX | Decisão de oferta de marketing no relacionamento banco–cliente, com controles de minimização e opt-out no canal |
| Cumprimento de obrigação legal | art. 7º, II | Retenção de trilha de auditoria de recomendações (quando existir log) |
| Revisão humana | art. 20 | Decisões sensíveis permanecem com humano no loop (aprovação de modelo / mudança de política); a saída é escolha de campanha, não efeito jurídico individual (crédito, limite, etc.) |

**Fora de escopo:** concessão de crédito, pricing, definição de limite ou qualquer decisão com efeito jurídico individual.

---

## 4. Minimização

Controles já presentes no código / desenho:

1. **ETL remove `duration`** (`src/datathon/etl/bank_marketing_primary.py` — `LEAKAGE_COLS = ["duration"]`).  
   Só se conhece a duração da ligação *depois* do contato; usá-la seria vazamento temporal.
2. **Contexto reduzido na decisão:** o modelo contextual opera em **12 células** (4 faixas etárias × 3 categorias de job), não no registro completo do cliente.
3. **API valida só os campos necessários** para `/recommend`: `age`, `job`, `marital`, `education`, `contact`, `campaign`. Campos extras não devem ser persistidos.
4. **Sem PII no modelo salvo:** `thompson_model.json` guarda apenas alphas/betas/trials por contexto.
5. **Braços de campanha relabeled, não sintéticos:** Email_Campaign/SMS_Alert/Call_Premium são
   segmentos reais (canal × campaign) com nome de negócio, documentado em
   `datathon.bandit.assign_arm` — não misturar com o exercício exploratório separado em
   `campaign_synthesis.py` (nunca executado no pipeline real), que de fato simula taxas
   hipotéticas e carrega seus próprios avisos de limitação.

### Princípio operacional

> Coletar / processar o **mínimo** necessário para escolher o arm e explicar a recomendação.  
> Em produção: preferir persistir `age_group` + `job_category` + `arm` + timestamp, **não** nome/CPF/saldo bruto.

---

## 5. Ciclo de retenção

| Artefato | Retenção (desenho) | Justificativa |
|---|---|---|
| Log de recomendações (quando implementado) | **90 dias**, depois exclusão ou anonimização | Alinhado a `docs/ARQUITETURA.md`; suficiente para auditoria operacional curta |
| Modelo ativo (`thompson_model.json`) | Enquanto a versão estiver ativa; versões antigas arquivadas | Reprodutibilidade e rollback |
| Golden set | Vida do repositório | Casos de teste versionados; sem PII |
| Parquet processado / sintético | Enquanto o projeto for mantido | Regenerável a partir do Kaggle + seeds |
| MLflow metrics | 2 anos (desenho) | Comparar experimentos |
| Request/response em memória da API | Não persistir além do necessário | Minimização |



---

## 6. Política de logs e telemetria

### O que um decision log **deve** conter (desenho-alvo)

| Campo | Motivo |
|---|---|
| `decision_id` (UUID) | Rastreabilidade sem PII |
| `timestamp` | Auditoria temporal |
| `age_group`, `job_category` | Contexto minimizado |
| `recommended_arm`, `arm_name` | Decisão tomada |
| `expected_conversion` | Transparência |
| `model_version` | Qual política gerou a decisão |
| `rationale` / `reason_codes` | Direito de explicação |

### O que **não** deve entrar no log

- Nome, CPF, telefone, e-mail  
- Payload completo se contiver campos além do contrato  
- Conteúdo de prompts de LLM (se houver assistente no futuro)

### Telemetria

- Preferir métricas agregadas (taxa de conversão por contexto, volume de `/recommend`)  
- Em nuvem (AWS/Azure): RBAC, sem buckets públicos, credenciais via IAM / Managed Identity (`docs/ARQUITETURA.md`)

---

## 7. Direitos do titular (desenho para operação real)

| Direito | Como o desenho atenderia |
|---|---|
| **Acesso** | Exportar linhas do log pelo `decision_id` (e join interno ao cliente só nos sistemas do controlador) |
| **Exclusão / anonimização** | Apagar ou anonimizar entradas do log antes do prazo de retenção; manter só agregados de modelo |
| **Explicação** | Campo `rationale` já retornado pela API; evoluir para reason codes estáveis |
| **Oposição / opt-out** | Opt-out no canal de contato (não ofertar campanha àquele canal) |
| **Revisão humana** | Mudança de modelo / política via processo com aprovação nomeada (humano no loop) |

---

## 8. Plano de resposta a incidentes

1. **Detectar** — alerta de acesso anômalo, report humano ou falha de permissão.  
2. **Conter** — revogar credenciais/roles; se necessário, derrubar a API (`fail closed`).  
3. **Avaliar** — classificar o dado exposto. No build atual: dado sintético / Kaggle anônimo → severidade limitada a “exposição de artefato interno”.  
4. **Notificar** — se houvesse dado pessoal real: DPO do controlador → ANPD e titulares no prazo razoável (LGPD art. 48), com registro do incidente.  
5. **Aprender** — correção via mudança versionada (novo modelo / nova política), documentada; não “consertar em silêncio” no modelo em produção.


---

## 9. Revisão e responsáveis

| Item | Definição |
|---|---|
| Dono do plano | Grupo 37 — Datathon FIAP |
| Quando revisar | A cada novo treino/publicação de modelo e no mínimo a cada entrega de etapa |
| Gatilho obrigatório | Qualquer mudança no schema de `/recommend`, no ETL ou no que é persistido |
| Evidência de aceite | Este arquivo versionado no repositório + checklist Etapa 8 atualizado |

---

## 10. Relação com outros documentos

| Documento | Papel |
|---|---|
| Este arquivo (`docs/LGPD_PLAN.md`) | Base legal, mapeamento, minimização, retenção, logs, incidentes |
| `docs/ARQUITETURA.md` | Arquitetura local/AWS/Azure; menção de retenção de 90 dias |
| `README.md` | Visão geral do problema e do algoritmo |
| `data/kaggle/*/README.md` | Fonte, licença e limitações de cada dataset |

---

## 11. Declaração final

Este sistema é uma **demonstração acadêmica**. Não processa clientes reais.  
A postura LGPD descrita acima é o **contrato de engenharia** do projeto: minimização estrutural, ausência de identificadores e atributos protegidos, finalidade única, retenção limitada e caminho claro para auditoria e resposta a incidentes.

