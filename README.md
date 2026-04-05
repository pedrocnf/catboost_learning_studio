# CatBoost Learning Studio

App didático em Streamlit para estudo de CatBoost, EDA, transformação, feature selection, treinamento, avaliação e interpretabilidade.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Assinatura

Prof. Pedro Nascimento  
GitHub: www.github.com/pedrocnf

## Deploy simples no Google Cloud

A forma mais simples e barata para este app é usar **Cloud Run** com `min-instances=0` e `max-instances=1`.

### Deploy manual

```bash
gcloud run deploy catboost-learning-studio \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

### Deploy com Terraform

1. Crie o repositório Artifact Registry e o serviço no Cloud Run com os arquivos em `terraform/`.
2. Build a imagem e envie para o Artifact Registry.
3. Aponte a variável `container_image` para a imagem publicada.

### GitHub Actions

O workflow em `.github/workflows/deploy-cloud-run.yml` usa:
- `google-github-actions/auth@v3`
- `google-github-actions/setup-gcloud@v3`
- `google-github-actions/deploy-cloudrun@v3`

Secrets esperados:
- `GCP_PROJECT_ID`
- `GCP_WIF_PROVIDER`
- `GCP_SERVICE_ACCOUNT`
