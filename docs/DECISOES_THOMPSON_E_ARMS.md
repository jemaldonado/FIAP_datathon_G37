# Decisões sobre Thompson Sampling e os arms

Este documento existe porque boa parte dessa explicação hoje só aparece rodando os dashboards
Streamlit (em especial a página "Thompson Aprende?" de `app_dashboard_pt.py`) — e quem for revisar
o projeto pelo repositório, sem subir os apps, não tem como ver isso. Aqui está por escrito.

## Por que Thompson Sampling, e não Epsilon-Greedy ou UCB

O edital permite qualquer um dos três (Thompson Sampling, Epsilon-Greedy, UCB), desde que a escolha
seja justificada. Escolhemos Thompson Sampling (Beta-Bernoulli) por três motivos concretos:

1. **Sem hiperparâmetro de exploração para ajustar.** Epsilon-Greedy precisa de um `epsilon` fixo
   (ou um schedule de decaimento) decidido a priori — errar esse valor custa conversão real: alto
   demais desperdiça tráfego no braço pior por tempo demais, baixo demais trava cedo no braço
   errado. Thompson Sampling não tem esse parâmetro: a exploração cai sozinha à medida que a
   posterior de cada braço fica mais estreita (mais `trials`), sem ninguém decidir uma taxa fixa.
2. **A incerteza é interpretável.** Cada braço tem uma distribuição Beta(`alpha`, `beta`)
   (`BetaBernoulliBandit` em `src/datathon/bandit/contextual_thompson.py`), inicializada
   uniforme (`alpha=1, beta=1`, ou seja, sem viés inicial nenhum). Poucos `trials` → posterior
   larga → mais exploração; muitos `trials` → posterior estreita → mais explotação. Isso é direto
   de mostrar e explicar; UCB também converge, mas o intervalo de confiança que ele usa é menos
   intuitivo de justificar para uma audiência de negócio do que "o modelo ainda não viu dado
   suficiente desse braço".
3. **Com 12 contextos e só 41 mil clientes, alguns contextos têm poucas amostras** (o contexto
   `Senior + Business` tem 96 clientes contra 6.689 de `Prime + Other`). Thompson Sampling se
   comporta bem nesse regime de poucos dados por construção — a posterior larga de um contexto
   com poucos trials já reflete a incerteza real, sem precisar de um epsilon separado por contexto.

Não avaliamos UCB/Epsilon-Greedy empiricamente lado a lado — a escolha foi de projeto, não uma
comparação experimental. Se fosse repetir, a comparação mais informativa seria contra Epsilon-Greedy
com um epsilon decrescente, no mesmo dado.

## Por que contextual (12 contextos), e não um bandit único

Um bandit único aprenderia "qual braço é melhor em média" — mas a conversão varia de 7,5% (pior
contexto) a 41,1% (melhor contexto), mais de 30 pontos percentuais de diferença. Uma política única
sem contexto nunca captura isso: ela convergiria para o braço melhor "na média geral", que pode ser
o pior braço para um contexto específico. Por isso `ContextualThompsonSampling` mantém 12 bandits
Beta-Bernoulli independentes — um por combinação de `age_group` (Young/Prime/Mature/Senior) ×
`job_category` (Technical/Business/Other) — cada um aprendendo a ordem de braços certa para aquele
segmento, sem interferir nos outros 11.

## A decisão de relabeling dos 4 arms

A base real (`bank-marketing`, UCI/Kaggle) só registra 2 canais de contato de verdade: `contact`
(`cellular` ou `telephone`). Não existem canais de e-mail, SMS ou "premium" nos dados. Para ter 4
braços — e não só 2 — de forma honesta (sem inventar taxa nenhuma), dividimos cada canal por
primeiro-contato vs. contato-repetido (`campaign == 1` ou não), e demos nome de negócio aos 4
segmentos resultantes:

| Arm | Segmento real | Origem |
|---|---|---|
| Cellular_Standard | cellular, primeiro contato | `contact=='cellular' & campaign==1` |
| Email_Campaign | cellular, contato repetido | `contact=='cellular' & campaign>1` |
| SMS_Alert | telephone, primeiro contato | `contact=='telephone' & campaign==1` |
| Call_Premium | telephone, contato repetido | `contact=='telephone' & campaign>1` |

Todo `y` (conversão) contado por braço é 100% observado nos dados — só o nome do braço é uma
convenção para dar a leitura de "4 estratégias de campanha" a uma base que só tem 2 canais reais.
A regra completa está em `assign_arm()` (`src/datathon/bandit/contextual_thompson.py`), com o
mesmo raciocínio documentado no código-fonte.

**Alternativa que descartamos:** gerar 4 braços com taxas sintéticas/estimadas (mais fácil de
programar, mais braços "de verdade" no nome). Descartamos porque contradiz o requisito do edital de
não usar dado inventado — e porque uma versão anterior deste projeto tentou isso e a simulação
resultante (Etapa 3) dava sempre a mesma melhoria "por construção", independente do braço escolhido
(ver seção de aprendizados abaixo).

## Aprendizados do processo

Nenhum destes bugs quebrava o código visivelmente — todos produziam um número que parecia
razoável até alguém ler a lógica de novo, não só o resultado.

- **A primeira versão da comparação baseline vs. Thompson (Etapa 3) estava matematicamente
  quebrada.** A recompensa simulada não dependia do braço escolhido pelo bandit, então baseline e
  Thompson davam sempre o mesmo número por construção — 0% de melhoria. Corrigido usando as taxas
  reais medidas por canal (`cellular` 14,74%, `telephone` 5,23%), o que deu a melhoria real de
  +3,70 p.p. (11,27% → 14,97%).
- **Um bug de aliasing corrompia o modelo de produção durante simulações de canary.** Sem um
  `baseline_model_path` explícito, a variável do "lado baseline" do canary apontava para o mesmo
  objeto Python que o modelo de produção — cada chamada simulada de canary alterava
  permanentemente o modelo real, sem nenhum erro visível. Corrigido copiando o modelo
  (`copy.deepcopy`) em vez de referenciá-lo.
- **Esse segundo bug só apareceu numa segunda rodada de revisão**, depois que o primeiro já tinha
  sido corrigido — a correção da simulação circular abriu caminho de código que expôs o aliasing.
  Isso mudou como revisamos o resto do projeto: rodar o código de novo depois de cada correção,
  não só ler o diff, porque o output "parece razoável" é exatamente o que esconde esse tipo de bug.

## Referências

- Implementação: [`src/datathon/bandit/contextual_thompson.py`](../src/datathon/bandit/contextual_thompson.py)
- Métricas completas: [`RELATORIO_TECNICO.md`](../RELATORIO_TECNICO.md), seções 3 e 4
- Golden set (5 perfis validados): [`README.md`](../README.md#golden-set--5-perfis-validados)
