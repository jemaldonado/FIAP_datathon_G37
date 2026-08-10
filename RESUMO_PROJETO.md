# Resumo do Projeto — Datathon FIAP G37

Recomendador de canal de contato bancário via Thompson Sampling contextual. Este resumo segue a
mesma ordem do [roteiro de apresentação](docs/ROTEIRO_APRESENTACAO.md) — serve como versão escrita
do que é mostrado no vídeo, para quem quiser revisar sem assistir.

## O problema

Um banco decide todo dia qual canal usar para contatar cada cliente elegível a uma oferta. Regra
fixa desperdiça tráfego em quem não converteria; testar tudo via A/B tradicional é lento demais
para reagir. A pergunta que o projeto responde: como decidir, cliente a cliente, com um algoritmo
que aprende sozinho qual abordagem funciona melhor para cada perfil?

## Dados e abordagem

Base pública `bank-marketing` (Kaggle/UCI): 41.188 contatos de telemarketing de um banco
português, alvo é se o cliente aceitou um depósito a prazo. A coluna `duration` foi removida do
modelo porque só é conhecida depois da ligação — usá-la seria vazamento de dados.

Como baseline, comparamos com a estratégia mais simples possível: manter o mix histórico real de
canais, sem olhar contexto nenhum — 11,27% de conversão. Contra isso, implementamos Thompson
Sampling contextual: 12 modelos, um para cada combinação de faixa etária (Young, Prime, Mature,
Senior) e categoria profissional (Technical, Business, Other), aprendendo a taxa real de cada
canal e convergindo para o melhor. Resultado: 14,97% — quase 4 pontos percentuais a mais que o
baseline.

| Métrica | Valor |
|---|---|
| Baseline → Thompson Sampling | 11,27% → 14,97% (+3,70 p.p.) |
| Melhor contexto | Senior + Other, 42,2% |
| Pior contexto | 8,7% |
| Testes automatizados | 60/60 passando |
| Golden set | 5 perfis validados |

## Como o modelo decide

Cada um dos 12 contextos mantém uma distribuição Beta por estratégia de contato. Para cada cliente
novo, o modelo amostra uma probabilidade de sucesso por estratégia, escolhe a de maior valor
amostrado, observa se converteu de verdade, e atualiza a distribuição. É esse mecanismo de
exploração/explotação que faz a conversão variar de 8,7% no pior contexto até 42,2% no melhor —
uma diferença que uma política única, sem contexto, nunca capturaria.

A base real só tem dois canais (`cellular`, `telephone`); os 4 "braços" do bandit são esses dois
canais divididos por primeiro-contato vs. contato-repetido, com nomes de negócio. Toda taxa de
conversão usada é medida direto da base — não é estimativa.

## API e demo

O modelo é servido via `POST /recommend`: recebe idade, profissão, estado civil etc., identifica o
contexto do cliente e devolve a estratégia recomendada, a conversão esperada e a justificativa. A
demo ao vivo (Swagger) mostra dois perfis diferentes recebendo recomendações diferentes.

## Canary deploy — um caso real de retreino

Modelos são retreinados com dados novos, e isso pode mudar a recomendação. Ordenamos o dataset
pela data real de contato e comparamos um snapshot treinado só com os primeiros 70% dos contatos
com o modelo final, treinado com a campanha completa. Para clientes jovens em cargos técnicos, o
snapshot antigo recomendava campanha por e-mail (10,72% de conversão esperada); com mais dados
reais o modelo passa a recomendar contato celular padrão (19,66%) — quase 1,8x maior, e troca a
oferta vencedora. É uma troca real, medida nos dados, não simulada.

É exatamente esse tipo de mudança que o canary deploy existe para validar: a API expõe a nova
versão a uma fração do tráfego, monitora se ela realmente converte melhor, e só promove para
100% dos clientes se os números confirmarem — em vez de aplicar o retreino cegamente.

## Validação

Validamos com o golden set de 5 perfis representativos, conferindo se a recomendação faz sentido
para cada um. 60 testes automatizados cobrem a API, o bandit e o canary deploy. Cada treino fica
registrado no MLflow, com os parâmetros do modelo e as métricas de baseline vs. Thompson.

## Arquitetura de nuvem

Para produção, a arquitetura é enxuta de propósito: dados e modelo (poucos KBs) em object storage
(S3 na AWS, Blob Storage na Azure, Cloud Storage na GCP), treino como função serverless, API e
dashboard como serviços PaaS gerenciados — sem clusters de treino pesados, porque o problema não
exige isso. As três nuvens estão detalhadas em `docs/architecture/` (AWS, AZURE, GCP), cada uma
com diagrama, seleção de serviços, canary deploy e custo com fonte oficial.

## Em resumo

Um bandit contextual que aprende com cada cliente, valida contra baseline, versiona experimentos
no MLflow e usa canary deploy para validar retreinos com segurança antes de impactar todos os
clientes.
