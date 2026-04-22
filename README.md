# Behavioral Signature Project

Pipeline experimental para análise de comportamento humano como assinatura digital, com foco na extração e organização de sinais comportamentais multimodais a partir de vídeos.

O objetivo é estruturar dados de forma reprodutível para permitir a análise de padrões e detecção de anomalias em contextos de risco.

---

## 📁 Estrutura do Projeto

```text
├── datasets/
│   ├── raw/          # Dados brutos (não versionados)
│   └── processed/    # Dados processados (não versionados)
├── scripts/
│   └── preprocess/   # Scripts de pré-processamento
├── outputs/          # Resultados e experimentos
└── config/           # Configurações (planejado)

## Ambiente

1. Criar e ativar ambiente virtual:

python3 -m venv venv
source venv/bin/activate

2. Instalar dependências:

pip install -r requirements.txt
Pré-processamento de vídeos

2.1 Para datasets baseados em vídeo:

```
python scripts/preprocess/preprocess_v2.py \
  --input_dir datasets/raw \
  --output_dir datasets/processed \
  --fps 5 \
  --max_videos 3
```

3. Pré-processamento de sequências de frames (ShanghaiTech)

Para datasets que já vêm em frames:

```
python scripts/preprocess/preprocess_frames.py \
  --input_dir datasets/shanghaitech/raw \
  --output_dir datasets/shanghaitech/processed \
  --stride 5 \
  --max_sequences 3
```

Saída esperada:

```
datasets/processed/
  frames/
    <sequence>/
      frame_000000.jpg
  metadata/
    metadata.csv
```

## Próximos passos

1. Validar preprocessamento em datasets reais
2. Mapear labels por frame (ShanghaiTech)
3. Extrair sinais comportamentais (movimento, postura, etc.)
4. Construir representação temporal do comportamento
5. Avaliar padrões e anomalias

## Observações
- Datasets não são versionados no repositório
- O foco atual é estruturar o pipeline de dados antes de avançar para modelagem