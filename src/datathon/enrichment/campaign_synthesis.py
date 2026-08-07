"""
Síntese de Campanhas Sintéticas com Thompson Sampling Contextual.

NÃO USADO EM PRODUÇÃO: este script é um exercício exploratório separado,
nunca executado pelo pipeline real (`data/synthetic_enrichment/` só tem
`.gitkeep`). O modelo servido pela API (`scripts/retrain_model.py` ->
`data/models/thompson_model.json`) NÃO usa as taxas simuladas aqui — os
4 braços de produção são segmentos reais (canal × campaign) relabeled,
com conversões 100% reais (ver `datathon.bandit.assign_arm`). Não citar
este arquivo como a origem da natureza "sintética" dos braços de produção
— eles não são sintéticos.

⚠️ AVISO CRÍTICO SOBRE LIMITAÇÕES:
═══════════════════════════════════════════════════════════════════════════════

Este arquivo cria dados SINTÉTICOS baseados em SUPOSIÇÕES sobre como diferentes
segmentos de clientes responderiam a campanhas que NUNCA foram testadas em
produção.

LIMITAÇÕES CONHECIDAS:
━━━━━━━━━━━━━━━━━━━━━━

1. DADOS NÃO OBSERVADOS:
   - Base Kaggle tem APENAS: Celular (14.74%), Telefone (5.23%)
   - NÃO tem: Email, SMS, Call Premium
   - Logo: Taxas para Email, SMS, Call Premium são ESTIMADAS, não observadas

2. SUPOSIÇÕES SÃO ARBITRÁRIAS:
   - Você supõe: "Young + SMS = 26.8%"
   - Base da suposição: "SMS é 1.5x melhor que celular"
   - Problema: Isso pode estar MUITO errado

   Exemplos de risco:
   ├─ Se SMS real = 5%: Você vai desperdiçar ofertas
   ├─ Se SMS real = 50%: Você perdeu oportunidade
   └─ Thompson levará 2-4 semanas para descobrir em produção

3. VIÉS NOS DADOS GERADOS:
   - Dados criados FAVORECEM suas suposições
   - Thompson aprende padrões que VOCÊ INVENTOU
   - Simulação pode ser MUITO diferente da produção

QUANDO USAR ESTE ARQUIVO:
━━━━━━━━━━━━━━━━━━━━━━

✓ ACEITÁVEL PARA:
  ├─ Datathon (projeto acadêmico)
  ├─ Prototipagem rápida
  ├─ Exploração de design space
  ├─ Demonstração de Thompson Sampling
  └─ Teste de pipeline de dados

✗ NÃO ACEITÁVEL PARA:
  ├─ Decisões de negócio críticas
  ├─ Alocação real de recursos
  ├─ Produção sem validação
  └─ Métricas de performance reais

FORMA CORRETA DE USAR:
━━━━━━━━━━━━━━━━━━━━

1. SIMULAÇÃO (o que você faz aqui):
   └─ Gera dados sintéticos com suposições
     └─ Útil para: explorar, testar pipeline, demonstrar conceito
     └─ Não é: previsão de produção

2. VALIDAÇÃO (o que você DEVE fazer):
   └─ A/B Testing em produção com amostra pequena
     └─ Ofereça SMS a 1.000 clientes Young reais
     └─ Meça taxa REAL
     └─ Compare com suposição (26.8%)
     └─ Ajuste modelo

3. PRODUÇÃO (com dados validados):
   └─ Use Thompson com taxas REAIS observadas
     └─ Thompson aprende adaptivamente
     └─ Sem viés (dados reais, não inventados)

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
from pathlib import Path
from typing import Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ContextualThompsonBandit:
    """
    Thompson Sampling que usa contexto do cliente.

    Mantém uma distribuição Beta separada para cada campanha,
    representando a incerteza sobre a taxa de conversão.
    """

    def __init__(self, n_campaigns: int = 6):
        self.n_campaigns = n_campaigns
        self.alpha = [1.0] * n_campaigns  # Sucessos iniciais
        self.beta = [1.0] * n_campaigns   # Fracassos iniciais
        self.history = []

    def select_campaign(self) -> int:
        """
        Escolher campanha usando Thompson Sampling.

        Algoritmo:
        1. Para cada campanha, amostrar taxa ~ Beta(alpha, beta)
        2. Retornar campanha com taxa amostrada mais alta
        """
        sampled_rates = [
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_campaigns)
        ]
        return np.argmax(sampled_rates)

    def observe_result(self, campaign_id: int, did_convert: bool) -> None:
        """Atualizar posterior após observar resultado."""
        if did_convert:
            self.alpha[campaign_id] += 1
        else:
            self.beta[campaign_id] += 1

        self.history.append({
            'campaign': campaign_id,
            'converted': did_convert,
            'alpha': self.alpha.copy(),
            'beta': self.beta.copy()
        })


def load_and_segment_data(kaggle_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Carregar dados do Kaggle e criar segmentos.

    Args:
        kaggle_path: Caminho para bank-additional-full.csv

    Returns:
        df: DataFrame com coluna 'segment'
        segments: Dict com metadados de segmentos
    """
    df = pd.read_csv(kaggle_path, sep=';')
    logger.info(f'✓ Carregado {len(df):,} clientes')

    # Segmentação por idade
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 30, 45, 60, 100],
        labels=['Young_<30', 'Prime_30-45', 'Mature_45-60', 'Senior_60+']
    )

    # Segmentação por profissão
    def categorize_job(job):
        job_lower = str(job).lower()
        if any(x in job_lower for x in ['admin', 'management', 'director']):
            return 'Admin'
        elif any(x in job_lower for x in ['technician', 'blue', 'craft']):
            return 'Professional'
        elif any(x in job_lower for x in ['services', 'housemaid']):
            return 'Services'
        elif any(x in job_lower for x in ['entrepreneur', 'self']):
            return 'Business'
        else:
            return 'Other'

    df['job_group'] = df['job'].apply(categorize_job)
    df['segment'] = df['age_group'].astype(str) + '_' + df['job_group'].astype(str)

    # Metadados de segmentos
    segments = {}
    for seg_name in df['segment'].unique():
        seg_data = df[df['segment'] == seg_name]
        segments[seg_name] = {
            'size': len(seg_data),
            'baseline_conversion': float((seg_data['y'] == 'yes').mean())
        }

    logger.info(f'✓ Criados {len(segments)} segmentos')
    return df, segments


def define_campaign_rates_by_segment(segments: Dict) -> Tuple[Dict, Dict]:
    """
    Definir taxa de conversão por segmento e campanha.

    ⚠️ CRÍTICO: Estas são ESTIMATIVAS, não observações!

    Cada multiplicador tem uma justificativa documentada,
    mas a realidade pode ser MUITO diferente.
    """

    campaigns = {
        0: {'name': 'Reference_Cellular_Call', 'base_rate': 0.1474},
        1: {'name': 'Telephone_Fixed', 'base_rate': 0.0523},
        2: {'name': 'Email_Generic', 'base_rate': 0.1032},
        3: {'name': 'Email_Personalized', 'base_rate': 0.7814},
        4: {'name': 'SMS_Alert', 'base_rate': 0.1789},
        5: {'name': 'Call_Premium', 'base_rate': 0.1394}
    }

    # Multiplicadores por segmento
    # ⚠️ AVISO: Baseados em conhecimento de domínio, NÃO validados
    multipliers_by_segment = {
        'Young_<30': {
            'Celular': 1.2,
            'Telefone': 0.5,
            'Email_Gen': 1.0,
            'Email_Pers': 0.9,
            'SMS': 1.5,  # ← CRÍTICO: SEM EVIDÊNCIA! Pode estar MUITO errado
            'Call_Prem': 0.7
        },
        'Prime_30-45': {
            'Celular': 1.0,
            'Telefone': 0.8,
            'Email_Gen': 1.1,
            'Email_Pers': 1.1,
            'SMS': 1.0,
            'Call_Prem': 0.9
        },
        'Mature_45-60': {
            'Celular': 1.1,
            'Telefone': 1.2,
            'Email_Gen': 0.9,
            'Email_Pers': 1.0,
            'SMS': 0.8,
            'Call_Prem': 1.1
        },
        'Senior_60+': {
            'Celular': 0.8,
            'Telefone': 1.5,
            'Email_Gen': 0.5,
            'Email_Pers': 0.7,
            'SMS': 0.3,  # ← CRÍTICO: SEM EVIDÊNCIA! Pode estar MUITO errado
            'Call_Prem': 1.3
        }
    }

    campaign_rates = {}

    for segment in segments.keys():
        age_group = '_'.join(segment.split('_')[:2])

        if age_group not in multipliers_by_segment:
            age_group = 'Prime_30-45'

        multipliers = multipliers_by_segment[age_group]
        campaign_rates[segment] = {}

        for campaign_id, campaign_info in campaigns.items():
            campaign_name = campaign_info['name']

            if 'Cellular' in campaign_name:
                mult = multipliers.get('Celular', 1.0)
            elif 'Telephone' in campaign_name:
                mult = multipliers.get('Telefone', 1.0)
            elif 'Email_Generic' in campaign_name:
                mult = multipliers.get('Email_Gen', 1.0)
            elif 'Email_Personalized' in campaign_name:
                mult = multipliers.get('Email_Pers', 1.0)
            elif 'SMS' in campaign_name:
                mult = multipliers.get('SMS', 1.0)
            elif 'Call_Premium' in campaign_name:
                mult = multipliers.get('Call_Prem', 1.0)
            else:
                mult = 1.0

            final_rate = min(0.95, campaign_info['base_rate'] * mult)
            campaign_rates[segment][campaign_id] = final_rate

    logger.info(f'✓ Definidas taxas para {len(campaign_rates)} segmentos')
    logger.warning('⚠️  Taxas são ESTIMADAS. Validar em produção com A/B testing.')
    return campaigns, campaign_rates


def simulate_thompson_contextual(
    df: pd.DataFrame,
    segments: Dict,
    campaigns: Dict,
    campaign_rates: Dict,
    output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simular Thompson Sampling Contextual.

    Para cada cliente:
    1. Identificar segmento
    2. Thompson (daquele segmento) escolhe campanha
    3. Simular conversão (usando taxa estimada)
    4. Thompson aprende resultado

    ⚠️ Importante: Em produção, as taxas REAIS podem ser diferentes!
    """

    thompson_by_segment = {
        seg: ContextualThompsonBandit(n_campaigns=6)
        for seg in segments.keys()
    }

    offer_events = []
    rewards = []

    logger.info(f'Simulando {len(df):,} clientes...')

    for idx, customer in df.iterrows():
        segment = customer['segment']
        thompson = thompson_by_segment[segment]
        campaign_id = thompson.select_campaign()
        conversion_prob = campaign_rates[segment][campaign_id]

        # Simular conversão
        did_convert = np.random.rand() < conversion_prob
        thompson.observe_result(campaign_id, did_convert)

        # Registrar evento
        offer_events.append({
            'customer_id': idx,
            'segment': segment,
            'age_group': customer['age_group'],
            'job_group': customer['job_group'],
            'campaign_id': campaign_id,
            'campaign_name': campaigns[campaign_id]['name'],
            'timestamp': pd.Timestamp.now(),
            'sent': True
        })

        # Registrar recompensa
        rewards.append({
            'customer_id': idx,
            'campaign_id': campaign_id,
            'converted': int(did_convert),
            'reward_delay_days': 7
        })

    logger.info(f'✓ Simulação completada')

    df_events = pd.DataFrame(offer_events)
    df_rewards = pd.DataFrame(rewards)

    # Salvar arquivos
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df_events.to_csv(f'{output_dir}/offer_events.csv', index=False)
    df_rewards.to_csv(f'{output_dir}/delayed_rewards.csv', index=False)

    logger.info(f'✓ Salvos offer_events.csv ({len(df_events):,} registros)')
    logger.info(f'✓ Salvos delayed_rewards.csv ({len(df_rewards):,} registros)')

    # Salvar catálogo com aviso crítico
    catalog = {
        'WARNING': 'Dados SINTÉTICOS com SUPOSIÇÕES. Validar em produção.',
        'campaigns': campaigns,
        'campaign_rates_by_segment': campaign_rates,
        'segments': segments
    }

    with open(f'{output_dir}/offer_catalog.json', 'w') as f:
        json.dump(catalog, f, indent=2, default=str)

    logger.info(f'✓ Salvos offer_catalog.json')

    _print_statistics(df_events, df_rewards, campaigns)

    return df_events, df_rewards


def _print_statistics(df_events: pd.DataFrame, df_rewards: pd.DataFrame, campaigns: Dict):
    """Imprimir estatísticas da simulação."""

    logger.info('\n' + '='*80)
    logger.info('ESTATÍSTICAS DA SIMULAÇÃO')
    logger.info('='*80)

    logger.info('\nDistribuição de ofertas por campanha:')
    for campaign_id, campaign_info in campaigns.items():
        count = (df_events['campaign_id'] == campaign_id).sum()
        pct = 100 * count / len(df_events)
        conversions = (df_rewards[
            (df_rewards['campaign_id'] == campaign_id) &
            (df_rewards['converted'] == 1)
        ]).shape[0]
        conv_rate = 100 * conversions / max(count, 1)
        logger.info(
            f'  {campaign_id}. {campaign_info["name"]:25} | '
            f'{count:6,} ({pct:5.1f}%) | '
            f'{conversions:5,} conversões ({conv_rate:5.1f}%)'
        )

    total_conversions = df_rewards['converted'].sum()
    total_customers = len(df_events)
    overall_rate = 100 * total_conversions / total_customers

    logger.info(f'\nTOTAL: {total_customers:,} clientes | '
                f'{total_conversions:,} conversões ({overall_rate:.2f}%)')

    logger.warning(
        '\n⚠️  LEMBRE-SE: Estas taxas são de SIMULAÇÃO com SUPOSIÇÕES.\n'
        '    Em produção, as taxas REAIS podem ser MUITO diferentes.\n'
        '    A/B testing é OBRIGATÓRIO para validar.'
    )


def main():
    """Orquestrar pipeline de síntese."""

    kaggle_path = 'data/kaggle/bank-marketing/bank-additional-full.csv'
    output_dir = 'data/synthetic_enrichment'

    logger.info('='*80)
    logger.info('CAMPAIGN SYNTHESIS - CONTEXTUAL THOMPSON SAMPLING')
    logger.info('='*80)
    logger.warning('\n⚠️  AVISO: Leia comentários no topo do arquivo sobre limitações.\n')

    logger.info('[1/3] Carregando e segmentando dados...')
    df, segments = load_and_segment_data(kaggle_path)

    logger.info('[2/3] Definindo taxas de conversão...')
    campaigns, campaign_rates = define_campaign_rates_by_segment(segments)

    logger.info('[3/3] Simulando Thompson Contextual...')
    simulate_thompson_contextual(df, segments, campaigns, campaign_rates, output_dir)

    logger.info('\n' + '='*80)
    logger.info('✓ PIPELINE CONCLUÍDA')
    logger.info('='*80)
    logger.info(f'\nArquivos criados em {output_dir}/')


if __name__ == '__main__':
    main()
